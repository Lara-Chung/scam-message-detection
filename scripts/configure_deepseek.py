"""Configure a local DeepSeek profile using a hidden native prompt and macOS Keychain."""
import getpass
import json
import os
import subprocess
from pathlib import Path

import yaml

from src.agent.local_credentials import CredentialError, read_keychain_secret, write_keychain_secret

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'ee6405.scam-message-detection.deepseek'


def main():
    account = getpass.getuser()
    secret = os.environ.get('LLM_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    if not secret:
        secret = read_keychain_secret(SERVICE, account)
    if not secret:
        # Captured stdout stays only in this process; it is never logged or returned.
        response = subprocess.run(['osascript', '-e', '''
            tell application "System Events"
                activate
                set answer to display dialog "Paste the DeepSeek API key to store it in macOS Keychain." default answer "" with hidden answer buttons {"Cancel", "Save"} default button "Save" with title "EE6405 DeepSeek Local Setup"
                return text returned of answer
            end tell
        '''], capture_output=True, text=True)
        if response.returncode:
            raise CredentialError('Local credential entry was cancelled or unavailable; no API request was made.')
        secret = response.stdout.rstrip('\r\n')
    write_keychain_secret(SERVICE, account, secret)
    if read_keychain_secret(SERVICE, account) != secret:
        raise CredentialError('Keychain round-trip failed.')
    profile = yaml.safe_load((ROOT/'configs/agent.yaml').read_text())
    profile['provider'].update(base_url='https://api.deepseek.com', model='deepseek-flash',
        keychain={'service':SERVICE, 'account':account})
    target=ROOT/'configs/agent.local.yaml'
    if target.is_symlink():
        raise CredentialError('Refusing a symlink configuration destination.')
    if target.exists():
        raise CredentialError('Local profile already exists; refusing to overwrite it.')
    with target.open('x') as stream:
        stream.write(yaml.safe_dump(profile, sort_keys=False))
    target.chmod(0o600)
    print(json.dumps({'credential_store':'macOS Keychain','key_configured':True,
        'profile':str(target),'model':'deepseek-flash','base_url':'https://api.deepseek.com'}))

if __name__ == '__main__':
    try:
        main()
    except CredentialError as error:
        print(json.dumps({'status':'configuration_pending','message':str(error)}))
        raise SystemExit(2)
