#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Fail if the hook library has more than one initializer (its own bootstrap constructor).
# In Wine preloader processes dyld never runs the initializers of DYLD_INSERT_LIBRARIES
# images, so any C++ static initializer would silently leave globals uninitialized there.
lib="$1"
status=0
for arch in $(lipo -archs "$lib"); do
    count=0
    # arm64/x86_64 with chained fixups: __TEXT,__init_offsets (4 bytes/entry)
    size=$(otool -arch "$arch" -l "$lib" | awk '/sectname __init_offsets/{getline; getline; getline; print $2}')
    [ -n "$size" ] && count=$((count + $(printf '%d' "$size") / 4))
    # legacy: __DATA*,__mod_init_func (8 bytes/entry)
    size=$(otool -arch "$arch" -l "$lib" | awk '/sectname __mod_init_func/{getline; getline; getline; print $2}')
    [ -n "$size" ] && count=$((count + $(printf '%d' "$size") / 8))
    if [ "$count" -ne 1 ]; then
        echo "error: $lib ($arch) has $count initializers, expected exactly 1." >&2
        echo "       Avoid namespace-scope objects with dynamic initialization (see hooks.mm)." >&2
        status=1
    fi
done
exit $status
