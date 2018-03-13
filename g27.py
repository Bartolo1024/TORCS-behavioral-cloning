#!/usr/bin/env python
"""
http://upgrayd.blogspot.de/2011/03/logitech-dual-action-usb-gamepad.html

G27
===
example::

    A0 B7 A3 04 5C 7D 02 02
     0  1  2  3  4  5  6  7


    0, 1, 2, 3: sequence, little endian
    3, 5: value, little endian
    6: group
    7: axis

NOTE! From here on, I talk big endian -- also in hex!

Wheel values::

            left               dead           right
            <-------------------XX---------------->
    dec     32769      65535     0      1     32767
    hex     80 01      ff ff  00 00 00 01     7F FF


Pedal values:

- no pressure: 7F FF
- halfway: 00 00
- full: 80 01
"""
from binascii import hexlify


def powergenerator(start=0):
    """Generate powers of 256"""
    i = start
    while True:
        yield 256 ** i
        i += 1


class Bytewurst(object):

    def __init__(self, bs):
        self.raw = bs
        self.ints = map(ord, bs)

    def __repr__(self):
        return ' '.join(map(hexlify, self.raw))

    @property
    def int(self):
        r"""
        For "01 00 03 0A" ints would be [1, 0, 3, 10], so::

            >>> bs = '\x01\x00\x03\x0A'
            >>> bw = Bytewurst(bs)
            >>> bw.int == (1 * 1) + (0 * 256) + (3 * 65536) + (10 * 16777216)
            True
        """
        return sum(a * b for a, b in zip(self.ints, powergenerator()))

    def hexLE(self):
        return hexlify(self.raw)

    @property
    def bits(self):
        return ' '.join([format(x, '08b') for x in self.ints])


BUTTON2NAME = """
0200=wheel axis
0105=paddle left
0104=paddle right
0107=wheel button left 1
0114=wheel button left 2
0115=wheel button left 3
0106=wheel button right 1
0112=wheel button right 2
0113=wheel button right 3
0201=clutch
0203=brake
0202=gas
0101=shifter button left
0102=shifter button right
0103=shifter button up
0100=shifter button down
0204=dpad left/right
0205=dpad up/down
010b=shifter button 1
0108=shifter button 2
0109=shifter button 3
010a=shifter button 4
010c=gear 1
010d=gear 2
010e=gear 3
010f=gear 4
0110=gear 5
0111=gear 6
0116=gear R
"""
button2namedict = dict(line.split('=') for line in BUTTON2NAME.strip().split('\n'))


class Button(Bytewurst):
    def __init__(self, bs):
        super(Button, self).__init__(bs)
        self.name = button2namedict.get(self.hexLE(), 'UNKNOWN: %s' % self.hexLE())


class Value(Bytewurst):
    def __repr__(self):
        if self.int == 0:
            return '  off'
        elif self.int == 1:
            return '   on'
        else:
            print(self.hexLE())
            return '%5d' % self.int


class Message(object):

    FMT_HEX = '%02X'
    FMT_DEC = '%03d'

    def __init__(self, bs):
        self.bs = bs
        self.raw_seq = bs[0:4]
        self.raw_value = bs[4:6]
        self.raw_id = bs[6:8]
        self.ints = map(ord, bs)
        self.sequence = Bytewurst(bs[0:4])
        self.value = Value(bs[4:6])
        self.button = Button(bs[6:8])

    def __repr__(self):
        values = (self.sequence.hexLE(), self.value, self.button.name)
        return str(self.sequence.hexLE()), str(self.value), str(self.button.name)
        #return '  '.join(map(str, values))

    @property
    def json(self):
        xs = (
            ('sequence', self.sequence.int),
            ('value', self.value),
            ('button', self.button),
        )
        #attrs = ('sequence', 'value', 'button')
        #xs = zip(attrs, (getattr(self, x) for x in attrs))
        return '{\n  ' + '\n  '.join('%s: %s' % x for x in xs) + '\n}'

    def hexLE(self):
        """
        Human-readable hex format. LITTLE ENDIAN!
        """
        return ' '.join(self.FMT_HEX % x for x in self.ints)

    @property
    def bit(self):
        return ' '.join([self.sequence.bits, self.value.bits, self.button.bits])

    @property
    def debug(self):
        self.button.hexLE()

    @property
    def bytewurst_hex(self):
        return '%s %s %s' % (self.sequence, self.value, self.button)

    @property
    def grouped_hex(self):
        return ' '.join(map(hexlify, (self.raw_seq, self.raw_value, self.raw_id)))

    @property
    def grouped_hex2(self):
        return '%02x %02x %02x' % (self.sequence, self.value, self.button)

    @property
    def fasthex(self):
        return hexlify(self.bs)

    @property
    def dec(self):
        """Human-readable decimal format"""
        return ' '.join(self.FMT_DEC % x for x in self.ints)


class WheelStateDetector(object):
    base_signals_dict = {
        'wheel axis': 'wheel',
        'clutch': 'clutch',
        'brake': 'brake',
        'gas': 'acceleration'
        }

    gear_dict = {
        'gear 1': 1,
        'gear 2': 2,
        'gear 3': 3,
        'gear 4': 4,
        'gear 5': 5,
        'gear 6': 6,
        'gear R': -1
    }

    paddle_dict = {
        'paddle left': -1,
        'paddle right': 1
    }

    other_buttons_dict = {
        'wheel button left 1': 0,
        'wheel button left 2': 0,
        'wheel button left 3': 0,
        'wheel button right 1': 0,
        'wheel button right 2': 0,
        'wheel button right 3': 0,
        'shifter button left': 0,
        'shifter button right': 0,
        'shifter button up': 0,
        'shifter button down': 0,
        'dpad left/right': 0,
        'dpad up/down': 0,
        'shifter button 1': 0,
        'shifter button 2': 0,
        'shifter button 3': 0,
        'shifter button 4': 0
    }

    def __init__(self)
        self.gear = 0
        self.angle = 0
        self.acceleration = 0
        self.brake = 0
        self.clutch = 0

    def dump_messages(self):
        with open('/dev/input/js0', 'rb') as device:
            while True:
                bs = device.read(8)
                hex, val, bt_name = Message(bs)
                self.decode_signal(hex, val, bt_name)
                #print(message)
                #message.debug

    def decode_signal(self, hex, value, bt_name):
        if bt_name == 'wheel axis':
            set_wheel(value)
        elif bt_name == 'clutch':
            set_clutch(value)
        elif bt_name == 'brake':
            set_brake(value)
        elif bt_name == 'gas':
            set_acceleration(value)
        elif bt_name in gear_dict:
            set_gear(value)

    def set_wheel(self, value):
        val = int(value)
        if val >= 32766 and val <= 65531: # zero
            self.angle = (65531 - 32766) / 32766
        elif val < 32766:
            self.angle = - val / 32766

    def set_acc(self):




if __name__ == '__main__':
    state = WheelStateDetector()
    state.dump_messages()
