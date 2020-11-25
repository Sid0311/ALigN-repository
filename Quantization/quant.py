import math
import tensorflow as tf

##########################################################################################
#These function take tensor values as arguments and return quantized tensor              #
##########################################################################################



# below function is for 8-bit linear quantization
def lin_8_bit(values):
  #values = tf.where(tf.is_nan(values), tf.zeros_like(values), values)
  bits = 8
  #print(type(values))
  range = 2**(bits-1)
  #print("actual value ")
  maxVal = tf.reduce_max(tf.abs(values))

  #print("max value " )
  step = (maxVal / range)

  step = tf.round(tf.log(step)/math.log(2))
  step = tf.cond(step>7.0, lambda: 7.0, lambda: step)
  step = tf.cond(step<-7.0, lambda: -7.0, lambda: step)
  step = tf.pow(2.0, step );
  qValues = tf.round(values / tf.cast(step,tf.float32))
  qValues = step * tf.clip_by_value(qValues,-range,range-1)
  return qValues
  
def lin_9_bit(values):
  bits = 9
  #values = tf.where(tf.is_nan(values), tf.zeros_like(values), values)
  #print(type(values))
  range = 2**(bits-1)
  #print("actual value ")
  maxVal = tf.reduce_max(tf.abs(values))

  #print("max value " )
  step = (maxVal / range)

  step = tf.round(tf.log(step)/math.log(2))
  step = tf.cond(step>8.0, lambda: 8.0, lambda: step)
  step = tf.cond(step<-8.0, lambda: -8.0, lambda: step)
  step = tf.pow(2.0, step );
  qValues = tf.round(values / tf.cast(step,tf.float32))
  qValues = step * tf.clip_by_value(qValues,-range,range-1)
  return qValues
	
	
##########################################################################################
# Below function are for align quantization
# ALigN_3_4 :- 3 bits for leading one location and 4 bits for following bits
# L2L is special configuration of ALigN i.e. ALigN_4_3
##########################################################################################

	
def align_4_3(tensor):
    #capturing the signs of tensor in the signs tensor
    signs=tf.sign(tensor)
	# get the learing one position
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    # for restricting the leading one till 16 bit
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -16.0, 16.0)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
	# residue1 is the value after removing the first leading bit
	# temp1 is first following bit after leading one
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    zero = temp1*-16.0
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))
    residue4 = residue3 % tf.pow(2*tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4/tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit))
    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + \
             tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3 + \
             temp3*temp4*(-tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3+ tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    tensor= tensor*signs
    return tensor

def align_6_1(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    # for restricting the leading one till 16 bit
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -64.0, 64.0)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)+tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + temp1*temp2*(-tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)*temp1+ tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit))
    tensor = tensor * signs
    return tensor

def align_3_4(tensor):
    signs=tf.sign(tensor)
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))

    residue4 = residue3 % tf.pow(2*tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4/tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit))

    residue5 = residue4 % tf.pow(2*tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5/tf.pow(2*tf.ones_like(sixth_lead_bit),sixth_lead_bit))

    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3 + tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit)*temp4+ temp4*temp5*(-tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit)*temp4+ tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))
    tensor= tensor*signs
    return tensor

def align_1_6(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -2, 2)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + temp6 * temp7 * (
                         -tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
                     2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))
    tensor = tensor * signs
    return tensor

def align_7_8(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -128, 128)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    ten_lead_bit = nine_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    residue9 = residue8 % tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit)
    temp9 = tf.floor(residue9 / tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
        2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7+ tf.pow(
        2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 + temp8 * temp9 * (
                         -tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 + tf.pow(
                     2 * tf.ones_like(eight_lead_bit), eight_lead_bit))
    tensor = tensor * signs
    return tensor

def align_1_14(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -2, 2)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    ten_lead_bit = nine_lead_bit - 1

    eleven_lead_bit = ten_lead_bit - 1
    twelve_lead_bit = eleven_lead_bit - 1
    thirteen_lead_bit = twelve_lead_bit - 1
    fourteen_lead_bit = thirteen_lead_bit - 1
    fifteen_lead_bit = fourteen_lead_bit - 1
    sixteen_lead_bit = fifteen_lead_bit - 1
    seventeen_lead_bit = sixteen_lead_bit - 1


    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    residue9 = residue8 % tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit)
    temp9 = tf.floor(residue9 / tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit))

    #######################################

    residue10 = residue9 % tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit)
    temp10 = tf.floor(residue10 / tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit))

    residue11 = residue10 % tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit)
    temp11 = tf.floor(residue11 / tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit))

    residue12 = residue11 % tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit)
    temp12 = tf.floor(residue12 / tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit))

    residue13 = residue12 % tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit)
    temp13 = tf.floor(residue13 / tf.pow(2 * tf.ones_like(fourteen_lead_bit), fourteen_lead_bit))

    residue14 = residue13 % tf.pow(2 * tf.ones_like(fourteen_lead_bit), fourteen_lead_bit)
    temp14 = tf.floor(residue14 / tf.pow(2 * tf.ones_like(fifteen_lead_bit), fifteen_lead_bit))

    residue15 = residue14 % tf.pow(2 * tf.ones_like(fifteen_lead_bit), fifteen_lead_bit)
    temp15 = tf.floor(residue15 / tf.pow(2 * tf.ones_like(sixteen_lead_bit), sixteen_lead_bit))

    residue16 = residue15 % tf.pow(2 * tf.ones_like(sixteen_lead_bit), sixteen_lead_bit)
    temp16 = tf.floor(residue16 / tf.pow(2 * tf.ones_like(seventeen_lead_bit), seventeen_lead_bit))

    #####################


    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
        2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7+ tf.pow(
        2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 +  tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit) * temp9 + tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit) * temp10 + tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit) * temp11 + tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit) * temp12 + tf.pow(2 * tf.ones_like(fourteen_lead_bit), fourteen_lead_bit) * temp13 +  tf.pow(2 * tf.ones_like(fifteen_lead_bit), fifteen_lead_bit) * temp14  + temp14 * temp15 * (
                         -tf.pow(2 * tf.ones_like(fifteen_lead_bit), fifteen_lead_bit) * temp14 + tf.pow(
                     2 * tf.ones_like(fourteen_lead_bit), fourteen_lead_bit))
    tensor = tensor * signs
    return tensor

def align_3_12(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    ten_lead_bit = nine_lead_bit - 1

    eleven_lead_bit = ten_lead_bit - 1
    twelve_lead_bit = eleven_lead_bit - 1
    thirteen_lead_bit = twelve_lead_bit - 1
    fourteen_lead_bit = thirteen_lead_bit - 1
    fifteen_lead_bit = fourteen_lead_bit - 1
    sixteen_lead_bit = fifteen_lead_bit - 1
    seventeen_lead_bit = sixteen_lead_bit - 1


    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    residue9 = residue8 % tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit)
    temp9 = tf.floor(residue9 / tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit))

    #######################################

    residue10 = residue9 % tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit)
    temp10 = tf.floor(residue10 / tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit))

    residue11 = residue10 % tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit)
    temp11 = tf.floor(residue11 / tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit))

    residue12 = residue11 % tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit)
    temp12 = tf.floor(residue12 / tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit))

    residue13 = residue12 % tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit)
    temp13 = tf.floor(residue13 / tf.pow(2 * tf.ones_like(fourteen_lead_bit), fourteen_lead_bit))

    #####################


    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
        2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7+ tf.pow(
        2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 +  tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit) * temp9 + tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit) * temp10 + tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit) * temp11 + tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit) * temp12  + temp12 * temp13 * (
                         -tf.pow(2 * tf.ones_like(thirteen_lead_bit), thirteen_lead_bit) * temp12 + tf.pow(
                     2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit))
    tensor = tensor * signs
    return tensor

def align_5_10(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -32, 32)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    ten_lead_bit = nine_lead_bit - 1

    eleven_lead_bit = ten_lead_bit - 1
    twelve_lead_bit = eleven_lead_bit - 1
    thirteen_lead_bit = twelve_lead_bit - 1
    fourteen_lead_bit = thirteen_lead_bit - 1
    fifteen_lead_bit = fourteen_lead_bit - 1
    sixteen_lead_bit = fifteen_lead_bit - 1
    seventeen_lead_bit = sixteen_lead_bit - 1


    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    residue9 = residue8 % tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit)
    temp9 = tf.floor(residue9 / tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit))

    #######################################

    residue10 = residue9 % tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit)
    temp10 = tf.floor(residue10 / tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit))

    residue11 = residue10 % tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit)
    temp11 = tf.floor(residue11 / tf.pow(2 * tf.ones_like(twelve_lead_bit), twelve_lead_bit))



    #####################


    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
        2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7+ tf.pow(
        2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 +  tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit) * temp9 + tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit) * temp10 + temp10 * temp11 * (
                         -tf.pow(2 * tf.ones_like(eleven_lead_bit), eleven_lead_bit) * temp10 + tf.pow(
                     2 * tf.ones_like(ten_lead_bit), ten_lead_bit))
    tensor = tensor * signs
    return tensor

def align_14_1(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    # for restricting the leading one till 14 bit
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -16384.0, 16384.0)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)+tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + temp1*temp2*(-tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)*temp1+ tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit))
    tensor = tensor * signs
    return tensor

def align_3_2(tensor):
    signs=tf.sign(tensor)
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))

    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + temp2*temp3*(-tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)*temp2+ tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    tensor= tensor*signs
    return tensor

def align_2_3(tensor):
    signs=tf.sign(tensor)
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -4, 4)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))

    residue4 = residue3 % tf.pow(2*tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4/tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit))

    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3 + temp3*temp4*(-tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3+ tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    tensor= tensor*signs
    return tensor

def align_3_3(tensor):
    signs=tf.sign(tensor)
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))

    residue4 = residue3 % tf.pow(2*tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4/tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit))

    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3 + temp3*temp4*(-tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3+ tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    tensor= tensor*signs
    return tensor

def align_2_4(tensor):
    signs=tf.sign(tensor)
    fir_lead_bit= tf.floor(tf.log(tf.abs(tensor+0.00000001))/math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -4, 4)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit)
    temp1 = tf.floor(residue1/tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit))
    residue2 = residue1 % tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit)
    temp2 = tf.floor(residue2/tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit))
    residue3 = residue2 % tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)
    temp3 = tf.floor(residue3/tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))

    residue4 = residue3 % tf.pow(2*tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4/tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit))

    residue5 = residue4 % tf.pow(2*tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5/tf.pow(2*tf.ones_like(sixth_lead_bit),sixth_lead_bit))

    tensor = tf.pow(2*tf.ones_like(fir_lead_bit),fir_lead_bit) + tf.pow(2*tf.ones_like(sec_lead_bit),sec_lead_bit) *temp1 + tf.pow(2*tf.ones_like(third_lead_bit),third_lead_bit)* temp2 + tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit)*temp3 + tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit)*temp4+ temp4*temp5*(-tf.pow(2*tf.ones_like(fifth_lead_bit),fifth_lead_bit)*temp4+ tf.pow(2*tf.ones_like(fourth_lead_bit),fourth_lead_bit))
    tensor= tensor*signs
    return tensor

def align_2_6(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -4, 4)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + temp6 * temp7 * (
                         -tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(
                     2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))
    tensor = tensor * signs
    return tensor

def align_3_5(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seven_lead_bit = sixth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seven_lead_bit), seven_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + temp5 * temp6 * (
                         -tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit) * temp5 + tf.pow(
                     2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))
    tensor = tensor * signs
    return tensor

def align_3_6(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -8, 8)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seven_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seven_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seven_lead_bit), seven_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seven_lead_bit), seven_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seven_lead_bit), seven_lead_bit) * temp6 + temp6 * temp7 * (
                         -tf.pow(2 * tf.ones_like(seven_lead_bit), seven_lead_bit) * temp6 + tf.pow(
                     2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))
    tensor = tensor * signs
    return tensor

def align_2_5(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -4, 4)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + temp5 * temp6 * (
                         -tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit) * temp5 + tf.pow(
                     2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))
    tensor = tensor * signs
    return tensor

def align_1_7(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -2, 2)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(2 * tf.ones_like(eight_lead_bit),
                                                                               eight_lead_bit) * temp7 + temp7 * temp8 * (
                         -tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7 + tf.pow(
                     2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))
    tensor = tensor * signs
    return tensor

def align_2_7(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -4, 4)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(2 * tf.ones_like(eight_lead_bit),
                                                                               eight_lead_bit) * temp7 + temp7 * temp8 * (
                         -tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit) * temp7 + tf.pow(
                     2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))
    tensor = tensor * signs
    return tensor

def align_1_8(tensor):
    signs = tf.sign(tensor)
    fir_lead_bit = tf.floor(tf.log(tf.abs(tensor + 0.00000001)) / math.log(2))
    fir_lead_bit = tf.clip_by_value(fir_lead_bit, -2, 2)
    sec_lead_bit = fir_lead_bit - 1
    third_lead_bit = sec_lead_bit - 1
    fourth_lead_bit = third_lead_bit - 1
    fifth_lead_bit = fourth_lead_bit - 1
    sixth_lead_bit = fifth_lead_bit - 1
    seventh_lead_bit = sixth_lead_bit - 1
    eight_lead_bit = seventh_lead_bit - 1
    nine_lead_bit = eight_lead_bit - 1
    ten_lead_bit = nine_lead_bit - 1
    residue1 = tf.abs(tensor) - tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit)
    temp1 = tf.floor(residue1 / tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit))
    residue2 = residue1 % tf.pow(2 * tf.ones_like(sec_lead_bit), sec_lead_bit)
    temp2 = tf.floor(residue2 / tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit))
    residue3 = residue2 % tf.pow(2 * tf.ones_like(third_lead_bit), third_lead_bit)
    temp3 = tf.floor(residue3 / tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit))

    residue4 = residue3 % tf.pow(2 * tf.ones_like(fourth_lead_bit), fourth_lead_bit)
    temp4 = tf.floor(residue4 / tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit))

    residue5 = residue4 % tf.pow(2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit)
    temp5 = tf.floor(residue5 / tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit))

    residue6 = residue5 % tf.pow(2 * tf.ones_like(sixth_lead_bit), sixth_lead_bit)
    temp6 = tf.floor(residue6 / tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit))

    residue7 = residue6 % tf.pow(2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit)
    temp7 = tf.floor(residue7 / tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit))

    residue8 = residue7 % tf.pow(2 * tf.ones_like(eight_lead_bit), eight_lead_bit)
    temp8 = tf.floor(residue8 / tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit))

    residue9 = residue8 % tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit)
    temp9 = tf.floor(residue9 / tf.pow(2 * tf.ones_like(ten_lead_bit), ten_lead_bit))

    tensor = tf.pow(2 * tf.ones_like(fir_lead_bit), fir_lead_bit) + tf.pow(2 * tf.ones_like(sec_lead_bit),
                                                                           sec_lead_bit) * temp1 + tf.pow(
        2 * tf.ones_like(third_lead_bit), third_lead_bit) * temp2 + tf.pow(2 * tf.ones_like(fourth_lead_bit),
                                                                           fourth_lead_bit) * temp3 + tf.pow(
        2 * tf.ones_like(fifth_lead_bit), fifth_lead_bit) * temp4 + tf.pow(2 * tf.ones_like(sixth_lead_bit),
                                                                           sixth_lead_bit) * temp5 + tf.pow(
        2 * tf.ones_like(seventh_lead_bit), seventh_lead_bit) * temp6 + tf.pow(2 * tf.ones_like(eight_lead_bit),
                                                                               eight_lead_bit) * temp7 + tf.pow(
        2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 + temp8 * temp9 * (
                         -tf.pow(2 * tf.ones_like(nine_lead_bit), nine_lead_bit) * temp8 + tf.pow(
                     2 * tf.ones_like(eight_lead_bit), eight_lead_bit))
    tensor = tensor * signs
    return tensor





