"""Explicit macOS Keychain storage. Secrets never travel through shell arguments."""
from __future__ import annotations

import ctypes
import platform


class CredentialError(RuntimeError):
    pass


def _security():
    if platform.system() != 'Darwin':
        raise CredentialError('macOS Keychain is unavailable on this platform.')
    security = ctypes.CDLL('/System/Library/Frameworks/Security.framework/Security')
    security.SecKeychainFindGenericPassword.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p,
        ctypes.c_uint32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    security.SecKeychainFindGenericPassword.restype = ctypes.c_int32
    security.SecKeychainAddGenericPassword.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p,
        ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    security.SecKeychainAddGenericPassword.restype = ctypes.c_int32
    security.SecKeychainItemModifyAttributesAndData.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_char_p]
    security.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32
    security.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    security.SecKeychainItemFreeContent.restype = ctypes.c_int32
    return security


def _find(security, service, account):
    service_bytes, account_bytes = service.encode(), account.encode()
    length, data, item = ctypes.c_uint32(), ctypes.c_void_p(), ctypes.c_void_p()
    status = security.SecKeychainFindGenericPassword(None, len(service_bytes), service_bytes,
        len(account_bytes), account_bytes, ctypes.byref(length), ctypes.byref(data), ctypes.byref(item))
    return status, length, data, item


def read_keychain_secret(service: str, account: str) -> str | None:
    security = _security()
    status, length, data, item = _find(security, service, account)
    if status == -25300:  # errSecItemNotFound
        return None
    if status:
        raise CredentialError(f'Keychain lookup failed (OSStatus {status}); no secret was printed.')
    try:
        return ctypes.string_at(data, length.value).decode()
    finally:
        security.SecKeychainItemFreeContent(None, data)
        ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation').CFRelease(ctypes.c_void_p(item.value))


def write_keychain_secret(service: str, account: str, secret: str):
    if not secret or any(c.isspace() for c in secret):
        raise CredentialError('An API key must be nonempty and contain no whitespace.')
    security = _security()
    status, length, data, item = _find(security, service, account)
    value = secret.encode()
    if status == 0:
        security.SecKeychainItemFreeContent(None, data)
        try:
            status = security.SecKeychainItemModifyAttributesAndData(item, None, len(value), value)
        finally:
            ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation').CFRelease(ctypes.c_void_p(item.value))
    elif status == -25300:
        service_bytes, account_bytes = service.encode(), account.encode()
        status = security.SecKeychainAddGenericPassword(None, len(service_bytes), service_bytes,
            len(account_bytes), account_bytes, len(value), value, None)
    if status:
        raise CredentialError(f'Keychain storage failed (OSStatus {status}); no secret was printed.')
