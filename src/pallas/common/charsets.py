ASCII_CHARSET = set("".join(chr(i) for i in range(128)))
EXTENDED_ASCII_CHARSET = set("".join(chr(i) for i in range(256)))

OCTAL_CHARSET = set("01234567")
DECIMAL_CHARSET = set("0123456789")

BASE64_CHARSET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')

# Hex charset is case-insensitive, so we only include lowercase
HEX_CHARSET = set('0123456789abcdef')