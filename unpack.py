import numpy as np

class UnPackage:
    """
    UnPackage类用于根据给定的数据格式和参数，对输入数据包进行解析和数据拆包。

    属性:
        data_bit (int): 基本数据位宽，默认为4。
        head_length (list): 包头长度列表，对应不同模式下的包头长度(Byte)。
        _sql_data_type_relate (dict): 数据类型相关信息映射表(针对SQL数据类型)。
        _common_data_type_relate (dict): 数据类型相关信息映射表(针对common数据类型)。
    """
    data_bit = 4
    head_length = [256, 32]  # Byte

    _sql_data_type_relate = {
        0: [np.uint16, 16, False],
        1: [np.int16, 16, False],
        2: [np.uint8, 8, False],
        3: [np.int8, 8, False],
        4: [np.uint32, 32, False],
        5: [np.int32, 32, False],
        6: [np.uint32, 32, True],  # complex类型, IQ int16组合
    }

    _common_data_type_relate = {
        4: [np.uint16, 16, False],
        5: [np.int16, 16, False],
        2: [np.uint8, 8, False],
        3: [np.int8, 8, False],
        0: [np.uint32, 32, False],
        1: [np.int32, 32, False],
        6: [np.uint32, 32, True],  # complex类型, IQ int16组合
        13: [np.int8, 8, True],
    }

    def __init__(self):
        pass

    @classmethod
    def common_solve_data(cls, data, step=0, qv_length=1024, head_tag=0x18EFDC01):
        try:
            chnl_position = None
            chnl_value_bit = 2
            pack_info = None
            pack_mode = 0
            first_num = 0

            # 根据包头标识解析包头信息
            head_info = cls.get_head_info(first_num, pack_mode, data, pack_info, chnl_position, chnl_value_bit,
                                          head_tag=head_tag)
            assert head_info is not None, "没有匹配的包头格式"
            head_index, head_list, head_list_length = head_info
            data = data[head_index:]

            # 解析包头字段
            unpacked_head = cls.get_pack_info(pack_mode, data, pack_info)
            (pack_length, mode, data_length, head_length, data_dtype, width, bit_con_byte,
             chnl_count, include_tail, is_complex) = unpacked_head

            cut_length = qv_length * bit_con_byte  # 截取长度以数据位宽为单位
            return (unpacked_head,
                    cls.unpack_data(mode, data, pack_length, chnl_count, data_dtype, width, bit_con_byte,
                                    cut_length, head_length, data_length, step, is_complex))
        except AssertionError as e:
            print(e)
            return {}
        except Exception as e:
            print(e)
            return {}

    @classmethod
    def get_head_info(cls, first_num, pack_mode, data, pack_info, chnl_position, chnl_value_bit, head_tag=0x18EFDC01):
        if pack_mode:
            head_list = [int(i, 16) for i in pack_info["headword"].split("-")]
        else:
            head_list = [head_tag]

        head_list_length = len(head_list)
        max_head_value = max(head_list)
        max_head_index = head_list.index(max_head_value)

        head_indexes = np.where(data == max_head_value)[0]
        for index in head_indexes:
            if head_list_length != 1:
                head_index = index - max_head_index
                check_data = data[head_index: index + (head_list_length - max_head_index)]
                if not (check_data == head_list).all():
                    continue
            else:
                head_index = index

            chnl_num = cls.get_chnl_number(pack_mode, data, chnl_position, chnl_value_bit, head_index)
            if chnl_num == first_num:
                return head_index, head_list, head_list_length
        else:
            return None  # 没有匹配的包头格式

    @classmethod
    def get_chnl_number(cls, pack_mode, data, chnl_position, chnl_value_bit, index=0):
        if pack_mode:
            start_index = index + chnl_position // cls.data_bit
            chnl_num = cls.get_index_data(data, start_index, chnl_position, chnl_value_bit)
        else:
            chnl_num = data[index + 4] >> 16 & 0xffff
        return chnl_num

    @classmethod
    def get_pack_num(cls, pack_mode, data, pack_info=None):
        if pack_mode:
            pack_num_index = pack_info["packnum"]
            pack_num = data[pack_num_index // cls.data_bit]
        else:
            pack_num = data[1]
        return pack_num

    @classmethod
    def get_pack_info(cls, pack_mode, data, pack_info=None):
        if pack_mode:
            pack_length_site = pack_info["packlength"]
            pack_length_bit = pack_info["packlengthbit"]
            start_index = pack_length_site // cls.data_bit
            pack_length = cls.get_index_data(data, start_index, pack_length_site, pack_length_bit)

            pack_length_unit = pack_info["packlengthunit"]
            pack_length *= pack_length_unit
            chnl_count = pack_info["count"]
            mode = pack_info["mode"]
            include_tail = pack_info["tail"]
            head_length = pack_info["headlength"]
            data_type = pack_info["bit"]  # 数据类型
            width = pack_info["width"]  # 交织力度 Byte
            lengthmode = pack_info["lengthmode"]
            data_dtype, bit, is_complex = cls._sql_data_type_relate.get(data_type, [np.uint32, 32, False])
        else:
            pack_length = data[2]
            mode_info = data[3] & 0xff
            # 解析打包模式
            if (mode_info >> 4) & 1:
                mode = 2  # 单包头交织打包
            else:
                mode = mode_info & 1  # 0: 交织打包; 1: 数据段打包;
            include_tail = (mode_info >> 1) & 1
            # 超长包头判断
            if (mode_info >> 3) & 1:
                head_length = 512
            else:
                head_length = cls.head_length[(mode_info >> 2) & 1]

            width = (data[3] >> 16) & 0xffff
            chnl_count = data[4] & 0xffff
            data_type = data[6] & 0xff
            lengthmode = True
            data_dtype, bit, is_complex = cls._common_data_type_relate.get(data_type, [np.uint32, 32, False])

        bit_con_byte = bit // 8
        assert not width % bit_con_byte, "交织力度*8不能整除数据位宽"

        # 根据lengthmode和include_tail计算实际data_length
        if lengthmode:
            if include_tail:
                data_length = pack_length - head_length * 2
            else:
                data_length = pack_length - head_length
        else:
            # 长度不包含包头
            data_length = pack_length
            if include_tail:
                pack_length += head_length * 2
            else:
                pack_length += head_length

        if mode == 2:
            # 单包头交织需要对data_length做整除处理
            data_length = (data_length - data_length % chnl_count) // chnl_count

        return (pack_length, mode, data_length, head_length, data_dtype, width, bit_con_byte,
                chnl_count, include_tail, is_complex)

    @classmethod
    def get_index_data(cls, data, start_index, total_size, total_bit):
        left_bit = total_size % cls.data_bit
        if cls.data_bit - left_bit < total_bit / 8:
            current_pack_length_bit = (cls.data_bit - left_bit) * 8
            next_pack_length_bit = total_bit - current_pack_length_bit
        else:
            current_pack_length_bit = total_bit
            next_pack_length_bit = 0

        if left_bit:
            result = (data[start_index] >> (left_bit * 8)) & (2 ** current_pack_length_bit - 1)
        else:
            result = data[start_index] & (2 ** current_pack_length_bit - 1)

        if next_pack_length_bit:
            result = (result << next_pack_length_bit) + (data[start_index + 1] & (2 ** next_pack_length_bit - 1))
        return result

    @classmethod
    def unpack_data(cls, mode, data, pack_length, chnl_count, data_dtype, width, bit_con_byte, cut_length, head_length,
                    data_length, step, is_complex):
        """
        根据解析得到的参数对数据进行最终的拆包和截取处理。

        参数：
            mode (int): 打包模式
            data (np.ndarray): 包含包头的原始数据
            pack_length (int): 包长度(bit为单位)
            chnl_count (int): 通道数
            data_dtype (np.dtype): 数据类型
            width (int): 交织力度(字节)
            bit_con_byte (int): 数据类型对应的字节数
            cut_length (int): 截取长度(字节数)
            head_length (int): 包头长度(bit为单位)
            data_length (int): 数据长度(字节)
            step (int): 抽取步长
            is_complex (bool): 是否复数类型

        返回:
            (head_data_list, data_array)
        """
        temp_head_length = head_length // bit_con_byte
        temp_pack_length = pack_length // bit_con_byte
        data_length //= bit_con_byte
        cut_length //= bit_con_byte
        head_length //= cls.data_bit
        pack_length //= cls.data_bit

        # 当step == -1时，根据数据长度与截取长度计算step
        if step == -1:
            step = data_length // cut_length
        else:
            step += 1

        total_cut_length = (cut_length - 1) * step + 1
        if total_cut_length > data_length:
            total_cut_length = data_length

        width_count = width // bit_con_byte
        if mode != 1:
            # 截取长度向上对齐到width_count的整数倍
            remainder = total_cut_length % width_count
            ceil_total_cut_length = total_cut_length + width_count - remainder if remainder else total_cut_length
        else:
            ceil_total_cut_length = total_cut_length

        step_index = np.arange(0, total_cut_length, step)

        data = data[: pack_length * chnl_count]
        if mode == 2:
            # 单包头交织打包
            head_data = data[: head_length]
            data = data[head_length:]
            data.dtype = data_dtype
            solve_data = data[: ceil_total_cut_length * chnl_count]
            assert len(solve_data) >= ceil_total_cut_length * chnl_count, "匹配的数据长度 < 单采集通道截取长度 * 采集通道数"
            data = solve_data.reshape(ceil_total_cut_length // width_count, chnl_count, width_count)
            data = np.einsum("abc->bac", data).reshape(chnl_count, ceil_total_cut_length)
        else:
            if mode == 1:
                # 数据段打包
                assert len(data) >= pack_length * chnl_count, "匹配的包数据长度 < 包长 * 采集通道数"
                head_data = data.reshape(chnl_count, pack_length)[..., : head_length]
                data.dtype = data_dtype
                data = data[: temp_pack_length * chnl_count].reshape(chnl_count, temp_pack_length)
                data = data[..., temp_head_length: ceil_total_cut_length + temp_head_length]
            else:
                # mode == 0 交织打包
                head_data = data[: head_length * chnl_count].reshape(head_length // (width // 4), chnl_count,
                                                                     width // 4)
                head_data = np.einsum("abc->bac", head_data).reshape(chnl_count, head_length)
                data.dtype = data_dtype
                solve_data = data[temp_head_length * chnl_count:
                                  (ceil_total_cut_length + temp_head_length) * chnl_count]
                assert len(solve_data) >= ceil_total_cut_length * chnl_count, "匹配的包数据长度 < (截取长度 + 包头长度) * 采集通道数"
                data = solve_data.reshape(ceil_total_cut_length // width_count, chnl_count, width_count)
                data = np.einsum("abc->bac", data).reshape(chnl_count, ceil_total_cut_length)

        data = data[..., step_index].astype("i8")

        if is_complex:
            data = cls.unpack_complex(data)

        return head_data.tolist(), data

    @classmethod
    def unpack_complex(cls, data):
        data = np.array(cls.unpack_complex_custom(data))
        return np.einsum("abc->bac", data)

    @classmethod
    def unpack_complex_custom(cls, data):
        real_part, imaginary_part = data & (2 ** 16 - 1), data >> 16
        real_part = real_part - 2 ** 16 * (real_part > 2 ** (16 - 1))
        imaginary_part = imaginary_part - 2 ** 16 * (imaginary_part > 2 ** (16 - 1))
        return real_part, imaginary_part
