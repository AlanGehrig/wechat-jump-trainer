"""Capture a previously verified physical screen rectangle into a top-down DIB."""
import ctypes as c
from ctypes import wintypes as w

from PIL import Image


class _BitmapInfo(c.Structure):
    _fields_ = [
        ("size", w.DWORD), ("width", w.LONG), ("height", w.LONG),
        ("planes", w.WORD), ("bits", w.WORD), ("compression", w.DWORD),
        ("image_size", w.DWORD), ("x_ppm", w.LONG), ("y_ppm", w.LONG),
        ("colors_used", w.DWORD), ("colors_important", w.DWORD),
        ("color_table", w.DWORD),
    ]


_user = c.WinDLL("user32", use_last_error=True)
_gdi = c.WinDLL("gdi32", use_last_error=True)


def _bind(dll, name, result, *arguments):
    fn = getattr(dll, name)
    fn.restype = result
    fn.argtypes = arguments
    return fn


_get_dc = _bind(_user, "GetDC", w.HDC, w.HWND)
_release = _bind(_user, "ReleaseDC", c.c_int, w.HWND, w.HDC)
_memory_dc = _bind(_gdi, "CreateCompatibleDC", w.HDC, w.HDC)
_delete_dc = _bind(_gdi, "DeleteDC", w.BOOL, w.HDC)
_create_dib = _bind(_gdi, "CreateDIBSection", w.HBITMAP, w.HDC,
                    c.POINTER(_BitmapInfo), w.UINT, c.POINTER(c.c_void_p), w.HANDLE, w.DWORD)
_select = _bind(_gdi, "SelectObject", w.HGDIOBJ, w.HDC, w.HGDIOBJ)
_delete = _bind(_gdi, "DeleteObject", w.BOOL, w.HGDIOBJ)
_copy = _bind(_gdi, "BitBlt", w.BOOL, w.HDC, c.c_int, c.c_int, c.c_int, c.c_int,
              w.HDC, c.c_int, c.c_int, w.DWORD)
_flush = _bind(_gdi, "GdiFlush", w.BOOL)


def grab_rect(bbox):
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("Empty capture area")
    screen = _get_dc(None)
    if not screen:
        raise c.WinError(c.get_last_error())
    memory = bitmap = previous = None
    try:
        memory = _memory_dc(screen)
        if not memory:
            raise c.WinError(c.get_last_error())
        info = _BitmapInfo(size=40, width=width, height=-height, planes=1, bits=32)
        bits = c.c_void_p()
        bitmap = _create_dib(screen, c.byref(info), 0, c.byref(bits), None, 0)
        if not bitmap or not bits.value:
            raise c.WinError(c.get_last_error())
        previous = _select(memory, bitmap)
        if not previous or previous == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        if not _copy(memory, 0, 0, width, height, screen, left, top, 0x00CC0020):
            raise c.WinError(c.get_last_error())
        _flush()
        raw = c.string_at(bits, width * height * 4)
        return Image.frombytes("RGB", (width, height), raw, "raw", "BGRX")
    finally:
        if previous and previous != c.c_void_p(-1).value:
            _select(memory, previous)
        if bitmap:
            _delete(bitmap)
        if memory:
            _delete_dc(memory)
        _release(None, screen)
