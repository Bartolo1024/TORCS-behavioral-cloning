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
from threading import Thread, Lock


def powergenerator(start=0):
    """Generate powers of 256"""
    i = start
    while True:
        yield 256 ** i
        i += 1


class Bytewurst(object):

    def __init__(self, bs):
        self.raw = bs
        self.ints = list(bytearray(bs))

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
            #print((self.hexLE()))
            return '%5d' % self.int


class Message(object):

    FMT_HEX = '%02X'
    FMT_DEC = '%03d'

    def __init__(self, bs):
        self.bs = bs
        self.raw_seq = bs[0:4]
        self.raw_value = bs[4:6]
        self.raw_id = bs[6:8]
        self.ints = list(bytearray(bs))
        self.sequence = Bytewurst(bs[0:4])
        self.value = Value(bs[4:6])
        self.button = Button(bs[6:8])

    def __repr__(self):
        values = (self.sequence.hexLE(), self.value, self.button.name)
        return '  '.join(map(str, values))

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


class WheelStateDetector(Thread):
    base_signals_dict = {
        'wheel axis': 'wheel',
        'clutch': 'clutch',
        'brake': 'brake',
        'gas': 'UNKNOWN: b\'0202\''
        }

    gear_dict = {
        'UNKNOWN: b\'010c\'': 1,
        'UNKNOWN: b\'010d\'': 2,
        'UNKNOWN: b\'010e\'': 3,
        'UNKNOWN: b\'010f\'': 4,
        'UNKNOWN: b\'010g\'': 5,
        'UNKNOWN: b\'010h\'': 6,
        'UNKNOWN: b\'010i\'': -1
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

    def __init__(self):
        self.is_active = True
        Thread.__init__(self)
        self.gear = 0
        self.angle = 0
        self.acceleration = 0
        self.brake = 0
        self.clutch = 0

    def run(self):
        self.dump_messages()
        #self.test_inf_loop()

    def dump_messages(self):
        with open('/dev/input/js0', 'rb') as device:
            while self.is_active:
                bs = device.read(8)
                message = Message(bs)
                self.decode_signal(message.sequence, message.value, message.button.name)
                #print(str(message.sequence) + str(message.value) + str(message.button.name))
                #print(message)
                message.debug

    def test_inf_loop(self):
        from g27_test.example_g27_data_preprocessing import TestInput
        ti = TestInput()
        while True:
            hex, val, bt_name = ti.get_random_signal()
            print(hex.decode("utf-8"), val, bt_name.decode("utf-8"))
            self.decode_signal(hex, val, bt_name)

    def decode_signal(self, hex, value, bt_name):
        #print(value)
        print('bt name is: % s and value is %s' % (bt_name, value))
        if str(bt_name) == 'UNKNOWN: b\'0200\'':
            self.set_wheel(str(value))
        elif str(bt_name) == 'UNKNOWN: b\'0201\'':
            self.set_clutch(str(value))
        elif str(bt_name) == 'UNKNOWN: b\'0203\'':
            self.set_brake(str(value))
        elif str(bt_name) == 'UNKNOWN: b\'0202\'':
            self.set_acceleration(str(value))
        elif str(bt_name) in self.gear_dict:
            self.set_gear(value)

    def set_wheel(self, value):
        print(value)
        try:
            self.angle = self.normalize_signal(int(value), zero_pivot=True)
            print(self.angle)
        except ValueError:
            self.angle = 0
            pass

    def set_acceleration(self, value):
        try:
            self.acceleration = self.normalize_signal(int(value), zero_pivot=False)
        except ValueError:
            self.acceleration = 0
            pass

    def set_brake(self, value):
        try:
            self.brake = self.normalize_signal(int(value), zero_pivot=False)
        except ValueError:
            self.brake = 0
            pass

    def set_clutch(self, value):
        try:
            self.clutch = self.normalize_signal(int(value), zero_pivot=False)
        except ValueError:
            self.clutch = 0
            pass

    @staticmethod
    def normalize_signal(value, zero_pivot=False):
        val = int(value)
        if zero_pivot is True:
            if val in range(32769, 65535):  # zero
                ret = (65535 - val) / 32769
            else:
                ret = - val / 32767
        else:
            if val <= 32767:
                ret = (32767 - val) / 65531
            else:
                ret = 0.5 + (65531 - val) / 65531
        return ret

    def set_gear(self, gear_shifter_position, off=False):
        if off is False:
            self.gear = 0
        elif gear_shifter_position in range(-1, 7):
            self.gear = int(gear_shifter_position)
        else:
            print('bad gear value')

    def get_action(self):
        return [self.angle, self.acceleration, self.gear, self.brake], self.clutch

    def stop(self):
        self.is_active = False


if __name__ == '__main__':
    state = WheelStateDetector()
    state.dump_messages()
