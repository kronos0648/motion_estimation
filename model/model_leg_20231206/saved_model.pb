��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��	
�
intermediate_weightsVarHandleOp*
_output_shapes
: *%

debug_nameintermediate_weights/*
dtype0*
shape:*%
shared_nameintermediate_weights
y
(intermediate_weights/Read/ReadVariableOpReadVariableOpintermediate_weights*
_output_shapes
:*
dtype0
�
false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
�
false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
�
true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
�
intermediate_weights_1VarHandleOp*
_output_shapes
: *'

debug_nameintermediate_weights_1/*
dtype0*
shape:*'
shared_nameintermediate_weights_1
}
*intermediate_weights_1/Read/ReadVariableOpReadVariableOpintermediate_weights_1*
_output_shapes
:*
dtype0
�
false_negatives_1VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_1/*
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
�
false_positives_1VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_1/*
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
�
true_positives_1VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_1/*
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
�
intermediate_weights_2VarHandleOp*
_output_shapes
: *'

debug_nameintermediate_weights_2/*
dtype0*
shape: *'
shared_nameintermediate_weights_2
y
*intermediate_weights_2/Read/ReadVariableOpReadVariableOpintermediate_weights_2*
_output_shapes
: *
dtype0
�
false_negatives_2VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_2/*
dtype0*
shape: *"
shared_namefalse_negatives_2
o
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes
: *
dtype0
�
false_positives_2VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_2/*
dtype0*
shape: *"
shared_namefalse_positives_2
o
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes
: *
dtype0
�
true_positives_2VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_2/*
dtype0*
shape: *!
shared_nametrue_positives_2
m
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
: *
dtype0
�
false_negatives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_3/*
dtype0*
shape:*"
shared_namefalse_negatives_3
s
%false_negatives_3/Read/ReadVariableOpReadVariableOpfalse_negatives_3*
_output_shapes
:*
dtype0
�
false_positives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_3/*
dtype0*
shape:*"
shared_namefalse_positives_3
s
%false_positives_3/Read/ReadVariableOpReadVariableOpfalse_positives_3*
_output_shapes
:*
dtype0
�
true_negativesVarHandleOp*
_output_shapes
: *

debug_nametrue_negatives/*
dtype0*
shape:*
shared_nametrue_negatives
m
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes
:*
dtype0
�
true_positives_3VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_3/*
dtype0*
shape:*!
shared_nametrue_positives_3
q
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes
:*
dtype0
�
false_negatives_4VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_4/*
dtype0*
shape:*"
shared_namefalse_negatives_4
s
%false_negatives_4/Read/ReadVariableOpReadVariableOpfalse_negatives_4*
_output_shapes
:*
dtype0
�
true_positives_4VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_4/*
dtype0*
shape:*!
shared_nametrue_positives_4
q
$true_positives_4/Read/ReadVariableOpReadVariableOptrue_positives_4*
_output_shapes
:*
dtype0
�
false_positives_4VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_4/*
dtype0*
shape:*"
shared_namefalse_positives_4
s
%false_positives_4/Read/ReadVariableOpReadVariableOpfalse_positives_4*
_output_shapes
:*
dtype0
�
true_positives_5VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_5/*
dtype0*
shape:*!
shared_nametrue_positives_5
q
$true_positives_5/Read/ReadVariableOpReadVariableOptrue_positives_5*
_output_shapes
:*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Nadam/v/dense_3/biasVarHandleOp*
_output_shapes
: *%

debug_nameNadam/v/dense_3/bias/*
dtype0*
shape:*%
shared_nameNadam/v/dense_3/bias
y
(Nadam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpNadam/v/dense_3/bias*
_output_shapes
:*
dtype0
�
Nadam/m/dense_3/biasVarHandleOp*
_output_shapes
: *%

debug_nameNadam/m/dense_3/bias/*
dtype0*
shape:*%
shared_nameNadam/m/dense_3/bias
y
(Nadam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpNadam/m/dense_3/bias*
_output_shapes
:*
dtype0
�
Nadam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *'

debug_nameNadam/v/dense_3/kernel/*
dtype0*
shape:	�*'
shared_nameNadam/v/dense_3/kernel
�
*Nadam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpNadam/v/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
Nadam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *'

debug_nameNadam/m/dense_3/kernel/*
dtype0*
shape:	�*'
shared_nameNadam/m/dense_3/kernel
�
*Nadam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpNadam/m/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
Nadam/v/dense_2/biasVarHandleOp*
_output_shapes
: *%

debug_nameNadam/v/dense_2/bias/*
dtype0*
shape:�*%
shared_nameNadam/v/dense_2/bias
z
(Nadam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpNadam/v/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Nadam/m/dense_2/biasVarHandleOp*
_output_shapes
: *%

debug_nameNadam/m/dense_2/bias/*
dtype0*
shape:�*%
shared_nameNadam/m/dense_2/bias
z
(Nadam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpNadam/m/dense_2/bias*
_output_shapes	
:�*
dtype0
�
Nadam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *'

debug_nameNadam/v/dense_2/kernel/*
dtype0*
shape:	�*'
shared_nameNadam/v/dense_2/kernel
�
*Nadam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpNadam/v/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
Nadam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *'

debug_nameNadam/m/dense_2/kernel/*
dtype0*
shape:	�*'
shared_nameNadam/m/dense_2/kernel
�
*Nadam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpNadam/m/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
Nadam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *&

debug_nameNadam/v/conv2d_3/bias/*
dtype0*
shape:*&
shared_nameNadam/v/conv2d_3/bias
{
)Nadam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpNadam/v/conv2d_3/bias*
_output_shapes
:*
dtype0
�
Nadam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *&

debug_nameNadam/m/conv2d_3/bias/*
dtype0*
shape:*&
shared_nameNadam/m/conv2d_3/bias
{
)Nadam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpNadam/m/conv2d_3/bias*
_output_shapes
:*
dtype0
�
Nadam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *(

debug_nameNadam/v/conv2d_3/kernel/*
dtype0*
shape:*(
shared_nameNadam/v/conv2d_3/kernel
�
+Nadam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpNadam/v/conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
Nadam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *(

debug_nameNadam/m/conv2d_3/kernel/*
dtype0*
shape:*(
shared_nameNadam/m/conv2d_3/kernel
�
+Nadam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpNadam/m/conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
Nadam/v/conv2d_2/biasVarHandleOp*
_output_shapes
: *&

debug_nameNadam/v/conv2d_2/bias/*
dtype0*
shape:*&
shared_nameNadam/v/conv2d_2/bias
{
)Nadam/v/conv2d_2/bias/Read/ReadVariableOpReadVariableOpNadam/v/conv2d_2/bias*
_output_shapes
:*
dtype0
�
Nadam/m/conv2d_2/biasVarHandleOp*
_output_shapes
: *&

debug_nameNadam/m/conv2d_2/bias/*
dtype0*
shape:*&
shared_nameNadam/m/conv2d_2/bias
{
)Nadam/m/conv2d_2/bias/Read/ReadVariableOpReadVariableOpNadam/m/conv2d_2/bias*
_output_shapes
:*
dtype0
�
Nadam/v/conv2d_2/kernelVarHandleOp*
_output_shapes
: *(

debug_nameNadam/v/conv2d_2/kernel/*
dtype0*
shape:*(
shared_nameNadam/v/conv2d_2/kernel
�
+Nadam/v/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpNadam/v/conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
Nadam/m/conv2d_2/kernelVarHandleOp*
_output_shapes
: *(

debug_nameNadam/m/conv2d_2/kernel/*
dtype0*
shape:*(
shared_nameNadam/m/conv2d_2/kernel
�
+Nadam/m/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpNadam/m/conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
Nadam/VariableVarHandleOp*
_output_shapes
: *

debug_nameNadam/Variable/*
dtype0*
shape: *
shared_nameNadam/Variable
i
"Nadam/Variable/Read/ReadVariableOpReadVariableOpNadam/Variable*
_output_shapes
: *
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:	�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	�*
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:	�*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�*
dtype0
�
conv2d_3/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_3/bias/*
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_3/kernel/*
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
�
conv2d_2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_2/bias/*
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_2/kernel/*
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
�
serving_default_conv2d_2_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_2_inputconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_524879

NoOpNoOp
�]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�]
value�]B�\ B�\
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator* 
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias*
<
0
1
 2
!3
54
65
D6
E7*
<
0
1
 2
!3
54
65
D6
E7*
* 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0
Ltrace_1* 

Mtrace_0
Ntrace_1* 
* 
�
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U
_u_product
V_update_step_xla*

Wserving_default* 

0
1*

0
1*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

 0
!1*

 0
!1*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

ktrace_0* 

ltrace_0* 
* 
* 
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

rtrace_0* 

strace_0* 

50
61*

50
61*
* 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

D0
E1*

D0
E1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
* 
* 
* 
* 
* 
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
WQ
VARIABLE_VALUENadam/Variable/optimizer/_u_product/.ATTRIBUTES/VARIABLE_VALUE*
r
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights*
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights*
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights*
b\
VARIABLE_VALUENadam/m/conv2d_2/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/v/conv2d_2/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/m/conv2d_2/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/v/conv2d_2/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/m/conv2d_3/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/v/conv2d_3/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/m/conv2d_3/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/v/conv2d_3/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/m/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/v/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/m/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/v/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/m/dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUENadam/v/dense_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/m/dense_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUENadam/v/dense_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_5=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_4>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_4=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_4>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
ga
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_3>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_3>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEintermediate_weights_2Ckeras_api/metrics/5/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEintermediate_weights_1Ckeras_api/metrics/6/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEintermediate_weightsCkeras_api/metrics/7/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	iterationlearning_rateNadam/VariableNadam/m/conv2d_2/kernelNadam/v/conv2d_2/kernelNadam/m/conv2d_2/biasNadam/v/conv2d_2/biasNadam/m/conv2d_3/kernelNadam/v/conv2d_3/kernelNadam/m/conv2d_3/biasNadam/v/conv2d_3/biasNadam/m/dense_2/kernelNadam/v/dense_2/kernelNadam/m/dense_2/biasNadam/v/dense_2/biasNadam/m/dense_3/kernelNadam/v/dense_3/kernelNadam/m/dense_3/biasNadam/v/dense_3/biastotal_1count_1totalcounttrue_positives_5false_positives_4true_positives_4false_negatives_4true_positives_3true_negativesfalse_positives_3false_negatives_3true_positives_2false_positives_2false_negatives_2intermediate_weights_2true_positives_1false_positives_1false_negatives_1intermediate_weights_1true_positivesfalse_positivesfalse_negativesintermediate_weightsConst*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_525375
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	iterationlearning_rateNadam/VariableNadam/m/conv2d_2/kernelNadam/v/conv2d_2/kernelNadam/m/conv2d_2/biasNadam/v/conv2d_2/biasNadam/m/conv2d_3/kernelNadam/v/conv2d_3/kernelNadam/m/conv2d_3/biasNadam/v/conv2d_3/biasNadam/m/dense_2/kernelNadam/v/dense_2/kernelNadam/m/dense_2/biasNadam/v/dense_2/biasNadam/m/dense_3/kernelNadam/v/dense_3/kernelNadam/m/dense_3/biasNadam/v/dense_3/biastotal_1count_1totalcounttrue_positives_5false_positives_4true_positives_4false_negatives_4true_positives_3true_negativesfalse_positives_3false_negatives_3true_positives_2false_positives_2false_negatives_2intermediate_weights_2true_positives_1false_positives_1false_negatives_1intermediate_weights_1true_positivesfalse_positivesfalse_negativesintermediate_weights*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_525537��
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_524682

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_524783
conv2d_2_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_524730o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524779:&"
 
_user_specified_name524777:&"
 
_user_specified_name524775:&"
 
_user_specified_name524773:&"
 
_user_specified_name524771:&"
 
_user_specified_name524769:&"
 
_user_specified_name524767:&"
 
_user_specified_name524765:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524939

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_525022

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_1_layer_call_fn_525010

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_524754a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
#__inference__update_step_xla_524904
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�
"
_user_specified_name
gradient
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_524754

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_524919
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�
W
#__inference__update_step_xla_524894
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:
"
_user_specified_name
gradient
�
�
(__inference_dense_3_layer_call_fn_525036

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_524723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name525032:&"
 
_user_specified_name525030:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_524694

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524670

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
W
#__inference__update_step_xla_524884
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:
"
_user_specified_name
gradient
�!
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_524762
conv2d_2_input)
conv2d_2_524733:
conv2d_2_524735:)
conv2d_3_524738:
conv2d_3_524740:!
dense_2_524745:	�
dense_2_524747:	�!
dense_3_524756:	�
dense_3_524758:
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_524733conv2d_2_524735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524654�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_524738conv2d_3_524740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524670�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524636�
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_524682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_524745dense_2_524747*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_524694�
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_524754�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_524756dense_3_524758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_524723w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:&"
 
_user_specified_name524758:&"
 
_user_specified_name524756:&"
 
_user_specified_name524747:&"
 
_user_specified_name524745:&"
 
_user_specified_name524740:&"
 
_user_specified_name524738:&"
 
_user_specified_name524735:&"
 
_user_specified_name524733:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�
�
)__inference_conv2d_3_layer_call_fn_524948

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524670w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524944:&"
 
_user_specified_name524942:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_524899
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_525047

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_524723

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
#__inference__update_step_xla_524914
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�
"
_user_specified_name
gradient
�
�
(__inference_dense_2_layer_call_fn_524989

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_524694p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524985:&"
 
_user_specified_name524983:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524636

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_525027

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�-
__inference__traced_save_525375
file_prefix@
&read_disablecopyonread_conv2d_2_kernel:4
&read_1_disablecopyonread_conv2d_2_bias:B
(read_2_disablecopyonread_conv2d_3_kernel:4
&read_3_disablecopyonread_conv2d_3_bias::
'read_4_disablecopyonread_dense_2_kernel:	�4
%read_5_disablecopyonread_dense_2_bias:	�:
'read_6_disablecopyonread_dense_3_kernel:	�3
%read_7_disablecopyonread_dense_3_bias:,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: 2
(read_10_disablecopyonread_nadam_variable: K
1read_11_disablecopyonread_nadam_m_conv2d_2_kernel:K
1read_12_disablecopyonread_nadam_v_conv2d_2_kernel:=
/read_13_disablecopyonread_nadam_m_conv2d_2_bias:=
/read_14_disablecopyonread_nadam_v_conv2d_2_bias:K
1read_15_disablecopyonread_nadam_m_conv2d_3_kernel:K
1read_16_disablecopyonread_nadam_v_conv2d_3_kernel:=
/read_17_disablecopyonread_nadam_m_conv2d_3_bias:=
/read_18_disablecopyonread_nadam_v_conv2d_3_bias:C
0read_19_disablecopyonread_nadam_m_dense_2_kernel:	�C
0read_20_disablecopyonread_nadam_v_dense_2_kernel:	�=
.read_21_disablecopyonread_nadam_m_dense_2_bias:	�=
.read_22_disablecopyonread_nadam_v_dense_2_bias:	�C
0read_23_disablecopyonread_nadam_m_dense_3_kernel:	�C
0read_24_disablecopyonread_nadam_v_dense_3_kernel:	�<
.read_25_disablecopyonread_nadam_m_dense_3_bias:<
.read_26_disablecopyonread_nadam_v_dense_3_bias:+
!read_27_disablecopyonread_total_1: +
!read_28_disablecopyonread_count_1: )
read_29_disablecopyonread_total: )
read_30_disablecopyonread_count: 8
*read_31_disablecopyonread_true_positives_5:9
+read_32_disablecopyonread_false_positives_4:8
*read_33_disablecopyonread_true_positives_4:9
+read_34_disablecopyonread_false_negatives_4:8
*read_35_disablecopyonread_true_positives_3:6
(read_36_disablecopyonread_true_negatives:9
+read_37_disablecopyonread_false_positives_3:9
+read_38_disablecopyonread_false_negatives_3:4
*read_39_disablecopyonread_true_positives_2: 5
+read_40_disablecopyonread_false_positives_2: 5
+read_41_disablecopyonread_false_negatives_2: :
0read_42_disablecopyonread_intermediate_weights_2: 8
*read_43_disablecopyonread_true_positives_1:9
+read_44_disablecopyonread_false_positives_1:9
+read_45_disablecopyonread_false_negatives_1:>
0read_46_disablecopyonread_intermediate_weights_1:6
(read_47_disablecopyonread_true_positives:7
)read_48_disablecopyonread_false_positives:7
)read_49_disablecopyonread_false_negatives:<
.read_50_disablecopyonread_intermediate_weights:
savev2_const
identity_103��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_2_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_2_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_nadam_variable"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_nadam_variable^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_nadam_m_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_nadam_m_conv2d_2_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_nadam_v_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_nadam_v_conv2d_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_nadam_m_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_nadam_m_conv2d_2_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_nadam_v_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_nadam_v_conv2d_2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_nadam_m_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_nadam_m_conv2d_3_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead1read_16_disablecopyonread_nadam_v_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp1read_16_disablecopyonread_nadam_v_conv2d_3_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_nadam_m_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_nadam_m_conv2d_3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_nadam_v_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_nadam_v_conv2d_3_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_nadam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_nadam_m_dense_2_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_nadam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_nadam_v_dense_2_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_nadam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_nadam_m_dense_2_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_nadam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_nadam_v_dense_2_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_nadam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_nadam_m_dense_3_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_nadam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_nadam_v_dense_3_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_nadam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_nadam_m_dense_3_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_nadam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_nadam_v_dense_3_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_total_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_28/DisableCopyOnReadDisableCopyOnRead!read_28_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp!read_28_disablecopyonread_count_1^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_total^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_30/DisableCopyOnReadDisableCopyOnReadread_30_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpread_30_disablecopyonread_count^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_31/DisableCopyOnReadDisableCopyOnRead*read_31_disablecopyonread_true_positives_5"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp*read_31_disablecopyonread_true_positives_5^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_false_positives_4"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_false_positives_4^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_33/DisableCopyOnReadDisableCopyOnRead*read_33_disablecopyonread_true_positives_4"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp*read_33_disablecopyonread_true_positives_4^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead+read_34_disablecopyonread_false_negatives_4"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp+read_34_disablecopyonread_false_negatives_4^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_35/DisableCopyOnReadDisableCopyOnRead*read_35_disablecopyonread_true_positives_3"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp*read_35_disablecopyonread_true_positives_3^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_36/DisableCopyOnReadDisableCopyOnRead(read_36_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp(read_36_disablecopyonread_true_negatives^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead+read_37_disablecopyonread_false_positives_3"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp+read_37_disablecopyonread_false_positives_3^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_38/DisableCopyOnReadDisableCopyOnRead+read_38_disablecopyonread_false_negatives_3"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp+read_38_disablecopyonread_false_negatives_3^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_39/DisableCopyOnReadDisableCopyOnRead*read_39_disablecopyonread_true_positives_2"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp*read_39_disablecopyonread_true_positives_2^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_40/DisableCopyOnReadDisableCopyOnRead+read_40_disablecopyonread_false_positives_2"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp+read_40_disablecopyonread_false_positives_2^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_41/DisableCopyOnReadDisableCopyOnRead+read_41_disablecopyonread_false_negatives_2"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp+read_41_disablecopyonread_false_negatives_2^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_intermediate_weights_2"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_intermediate_weights_2^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_43/DisableCopyOnReadDisableCopyOnRead*read_43_disablecopyonread_true_positives_1"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp*read_43_disablecopyonread_true_positives_1^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead+read_44_disablecopyonread_false_positives_1"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp+read_44_disablecopyonread_false_positives_1^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead+read_45_disablecopyonread_false_negatives_1"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp+read_45_disablecopyonread_false_negatives_1^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_intermediate_weights_1"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_intermediate_weights_1^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_47/DisableCopyOnReadDisableCopyOnRead(read_47_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp(read_47_disablecopyonread_true_positives^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_48/DisableCopyOnReadDisableCopyOnRead)read_48_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp)read_48_disablecopyonread_false_positives^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_49/DisableCopyOnReadDisableCopyOnRead)read_49_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp)read_49_disablecopyonread_false_negatives^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_intermediate_weights"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_intermediate_weights^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB/optimizer/_u_product/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/5/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/6/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/7/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *B
dtypes8
624	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_102Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_103IdentityIdentity_102:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_103Identity_103:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=49

_output_shapes
: 

_user_specified_nameConst:430
.
_user_specified_nameintermediate_weights:/2+
)
_user_specified_namefalse_negatives:/1+
)
_user_specified_namefalse_positives:.0*
(
_user_specified_nametrue_positives:6/2
0
_user_specified_nameintermediate_weights_1:1.-
+
_user_specified_namefalse_negatives_1:1--
+
_user_specified_namefalse_positives_1:0,,
*
_user_specified_nametrue_positives_1:6+2
0
_user_specified_nameintermediate_weights_2:1*-
+
_user_specified_namefalse_negatives_2:1)-
+
_user_specified_namefalse_positives_2:0(,
*
_user_specified_nametrue_positives_2:1'-
+
_user_specified_namefalse_negatives_3:1&-
+
_user_specified_namefalse_positives_3:.%*
(
_user_specified_nametrue_negatives:0$,
*
_user_specified_nametrue_positives_3:1#-
+
_user_specified_namefalse_negatives_4:0",
*
_user_specified_nametrue_positives_4:1!-
+
_user_specified_namefalse_positives_4:0 ,
*
_user_specified_nametrue_positives_5:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:40
.
_user_specified_nameNadam/v/dense_3/bias:40
.
_user_specified_nameNadam/m/dense_3/bias:62
0
_user_specified_nameNadam/v/dense_3/kernel:62
0
_user_specified_nameNadam/m/dense_3/kernel:40
.
_user_specified_nameNadam/v/dense_2/bias:40
.
_user_specified_nameNadam/m/dense_2/bias:62
0
_user_specified_nameNadam/v/dense_2/kernel:62
0
_user_specified_nameNadam/m/dense_2/kernel:51
/
_user_specified_nameNadam/v/conv2d_3/bias:51
/
_user_specified_nameNadam/m/conv2d_3/bias:73
1
_user_specified_nameNadam/v/conv2d_3/kernel:73
1
_user_specified_nameNadam/m/conv2d_3/kernel:51
/
_user_specified_nameNadam/v/conv2d_2/bias:51
/
_user_specified_nameNadam/m/conv2d_2/bias:73
1
_user_specified_nameNadam/v/conv2d_2/kernel:73
1
_user_specified_nameNadam/m/conv2d_2/kernel:.*
(
_user_specified_nameNadam/Variable:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_2/bias:/+
)
_user_specified_nameconv2d_2/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524969

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_525000

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_1_layer_call_fn_525005

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_524711p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_524928

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524654w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524924:&"
 
_user_specified_name524922:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524959

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_525537
file_prefix:
 assignvariableop_conv2d_2_kernel:.
 assignvariableop_1_conv2d_2_bias:<
"assignvariableop_2_conv2d_3_kernel:.
 assignvariableop_3_conv2d_3_bias:4
!assignvariableop_4_dense_2_kernel:	�.
assignvariableop_5_dense_2_bias:	�4
!assignvariableop_6_dense_3_kernel:	�-
assignvariableop_7_dense_3_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: ,
"assignvariableop_10_nadam_variable: E
+assignvariableop_11_nadam_m_conv2d_2_kernel:E
+assignvariableop_12_nadam_v_conv2d_2_kernel:7
)assignvariableop_13_nadam_m_conv2d_2_bias:7
)assignvariableop_14_nadam_v_conv2d_2_bias:E
+assignvariableop_15_nadam_m_conv2d_3_kernel:E
+assignvariableop_16_nadam_v_conv2d_3_kernel:7
)assignvariableop_17_nadam_m_conv2d_3_bias:7
)assignvariableop_18_nadam_v_conv2d_3_bias:=
*assignvariableop_19_nadam_m_dense_2_kernel:	�=
*assignvariableop_20_nadam_v_dense_2_kernel:	�7
(assignvariableop_21_nadam_m_dense_2_bias:	�7
(assignvariableop_22_nadam_v_dense_2_bias:	�=
*assignvariableop_23_nadam_m_dense_3_kernel:	�=
*assignvariableop_24_nadam_v_dense_3_kernel:	�6
(assignvariableop_25_nadam_m_dense_3_bias:6
(assignvariableop_26_nadam_v_dense_3_bias:%
assignvariableop_27_total_1: %
assignvariableop_28_count_1: #
assignvariableop_29_total: #
assignvariableop_30_count: 2
$assignvariableop_31_true_positives_5:3
%assignvariableop_32_false_positives_4:2
$assignvariableop_33_true_positives_4:3
%assignvariableop_34_false_negatives_4:2
$assignvariableop_35_true_positives_3:0
"assignvariableop_36_true_negatives:3
%assignvariableop_37_false_positives_3:3
%assignvariableop_38_false_negatives_3:.
$assignvariableop_39_true_positives_2: /
%assignvariableop_40_false_positives_2: /
%assignvariableop_41_false_negatives_2: 4
*assignvariableop_42_intermediate_weights_2: 2
$assignvariableop_43_true_positives_1:3
%assignvariableop_44_false_positives_1:3
%assignvariableop_45_false_negatives_1:8
*assignvariableop_46_intermediate_weights_1:0
"assignvariableop_47_true_positives:1
#assignvariableop_48_false_positives:1
#assignvariableop_49_false_negatives:6
(assignvariableop_50_intermediate_weights:
identity_52��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB/optimizer/_u_product/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/5/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/6/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/7/intermediate_weights/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_nadam_variableIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_nadam_m_conv2d_2_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_nadam_v_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_nadam_m_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_nadam_v_conv2d_2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_nadam_m_conv2d_3_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_nadam_v_conv2d_3_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_nadam_m_conv2d_3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_nadam_v_conv2d_3_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_nadam_m_dense_2_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_nadam_v_dense_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_m_dense_2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_nadam_v_dense_2_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_nadam_m_dense_3_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_nadam_v_dense_3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_nadam_m_dense_3_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_nadam_v_dense_3_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp$assignvariableop_31_true_positives_5Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_false_positives_4Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp$assignvariableop_33_true_positives_4Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_false_negatives_4Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp$assignvariableop_35_true_positives_3Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp"assignvariableop_36_true_negativesIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_false_positives_3Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_false_negatives_3Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp$assignvariableop_39_true_positives_2Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp%assignvariableop_40_false_positives_2Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp%assignvariableop_41_false_negatives_2Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_intermediate_weights_2Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp$assignvariableop_43_true_positives_1Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_false_positives_1Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp%assignvariableop_45_false_negatives_1Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_intermediate_weights_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp"assignvariableop_47_true_positivesIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp#assignvariableop_48_false_positivesIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp#assignvariableop_49_false_negativesIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_intermediate_weightsIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_52Identity_52:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:430
.
_user_specified_nameintermediate_weights:/2+
)
_user_specified_namefalse_negatives:/1+
)
_user_specified_namefalse_positives:.0*
(
_user_specified_nametrue_positives:6/2
0
_user_specified_nameintermediate_weights_1:1.-
+
_user_specified_namefalse_negatives_1:1--
+
_user_specified_namefalse_positives_1:0,,
*
_user_specified_nametrue_positives_1:6+2
0
_user_specified_nameintermediate_weights_2:1*-
+
_user_specified_namefalse_negatives_2:1)-
+
_user_specified_namefalse_positives_2:0(,
*
_user_specified_nametrue_positives_2:1'-
+
_user_specified_namefalse_negatives_3:1&-
+
_user_specified_namefalse_positives_3:.%*
(
_user_specified_nametrue_negatives:0$,
*
_user_specified_nametrue_positives_3:1#-
+
_user_specified_namefalse_negatives_4:0",
*
_user_specified_nametrue_positives_4:1!-
+
_user_specified_namefalse_positives_4:0 ,
*
_user_specified_nametrue_positives_5:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:40
.
_user_specified_nameNadam/v/dense_3/bias:40
.
_user_specified_nameNadam/m/dense_3/bias:62
0
_user_specified_nameNadam/v/dense_3/kernel:62
0
_user_specified_nameNadam/m/dense_3/kernel:40
.
_user_specified_nameNadam/v/dense_2/bias:40
.
_user_specified_nameNadam/m/dense_2/bias:62
0
_user_specified_nameNadam/v/dense_2/kernel:62
0
_user_specified_nameNadam/m/dense_2/kernel:51
/
_user_specified_nameNadam/v/conv2d_3/bias:51
/
_user_specified_nameNadam/m/conv2d_3/bias:73
1
_user_specified_nameNadam/v/conv2d_3/kernel:73
1
_user_specified_nameNadam/m/conv2d_3/kernel:51
/
_user_specified_nameNadam/v/conv2d_2/bias:51
/
_user_specified_nameNadam/m/conv2d_2/bias:73
1
_user_specified_nameNadam/v/conv2d_2/kernel:73
1
_user_specified_nameNadam/m/conv2d_2/kernel:.*
(
_user_specified_nameNadam/Variable:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:-)
'
_user_specified_nameconv2d_2/bias:/+
)
_user_specified_nameconv2d_2/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
F
*__inference_flatten_1_layer_call_fn_524974

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_524682`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_524980

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_524730
conv2d_2_input)
conv2d_2_524655:
conv2d_2_524657:)
conv2d_3_524671:
conv2d_3_524673:!
dense_2_524695:	�
dense_2_524697:	�!
dense_3_524724:	�
dense_3_524726:
identity�� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_524655conv2d_2_524657*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524654�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_524671conv2d_3_524673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524670�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524636�
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_524682�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_524695dense_2_524697*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_524694�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_524711�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_524724dense_3_524726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_524723w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:&"
 
_user_specified_name524726:&"
 
_user_specified_name524724:&"
 
_user_specified_name524697:&"
 
_user_specified_name524695:&"
 
_user_specified_name524673:&"
 
_user_specified_name524671:&"
 
_user_specified_name524657:&"
 
_user_specified_name524655:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�
�
-__inference_sequential_1_layer_call_fn_524804
conv2d_2_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_524762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524800:&"
 
_user_specified_name524798:&"
 
_user_specified_name524796:&"
 
_user_specified_name524794:&"
 
_user_specified_name524792:&"
 
_user_specified_name524790:&"
 
_user_specified_name524788:&"
 
_user_specified_name524786:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�5
�
!__inference__wrapped_model_524631
conv2d_2_inputN
4sequential_1_conv2d_2_conv2d_readvariableop_resource:C
5sequential_1_conv2d_2_biasadd_readvariableop_resource:N
4sequential_1_conv2d_3_conv2d_readvariableop_resource:C
5sequential_1_conv2d_3_biasadd_readvariableop_resource:F
3sequential_1_dense_2_matmul_readvariableop_resource:	�C
4sequential_1_dense_2_biasadd_readvariableop_resource:	�F
3sequential_1_dense_3_matmul_readvariableop_resource:	�B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity��,sequential_1/conv2d_2/BiasAdd/ReadVariableOp�+sequential_1/conv2d_2/Conv2D/ReadVariableOp�,sequential_1/conv2d_3/BiasAdd/ReadVariableOp�+sequential_1/conv2d_3/Conv2D/ReadVariableOp�+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOp�
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_1/conv2d_2/Conv2DConv2Dconv2d_2_input3sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_1/conv2d_2/ReluRelu&sequential_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:����������
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_1/conv2d_3/Conv2DConv2D(sequential_1/conv2d_2/Relu:activations:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential_1/flatten_1/ReshapeReshape-sequential_1/max_pooling2d_1/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:����������
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_1/dropout_1/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�
K
#__inference__update_step_xla_524889
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�
L
#__inference__update_step_xla_524909
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524654

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_524711

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_524879
conv2d_2_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_524631o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name524875:&"
 
_user_specified_name524873:&"
 
_user_specified_name524871:&"
 
_user_specified_name524869:&"
 
_user_specified_name524867:&"
 
_user_specified_name524865:&"
 
_user_specified_name524863:&"
 
_user_specified_name524861:_ [
/
_output_shapes
:���������
(
_user_specified_nameconv2d_2_input
�
L
0__inference_max_pooling2d_1_layer_call_fn_524964

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524636�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Q
conv2d_2_input?
 serving_default_conv2d_2_input:0���������;
dense_30
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias
 "_jit_compiled_convolution_op"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias"
_tf_keras_layer
X
0
1
 2
!3
54
65
D6
E7"
trackable_list_wrapper
X
0
1
 2
!3
54
65
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_0
Ltrace_12�
-__inference_sequential_1_layer_call_fn_524783
-__inference_sequential_1_layer_call_fn_524804�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0zLtrace_1
�
Mtrace_0
Ntrace_12�
H__inference_sequential_1_layer_call_and_return_conditional_losses_524730
H__inference_sequential_1_layer_call_and_return_conditional_losses_524762�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0zNtrace_1
�B�
!__inference__wrapped_model_524631conv2d_2_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U
_u_product
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
)__inference_conv2d_2_layer_call_fn_524928�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�
^trace_02�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524939�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0
):'2conv2d_2/kernel
:2conv2d_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_02�
)__inference_conv2d_3_layer_call_fn_524948�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0
�
etrace_02�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524959�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0
):'2conv2d_3/kernel
:2conv2d_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
ktrace_02�
0__inference_max_pooling2d_1_layer_call_fn_524964�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0
�
ltrace_02�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524969�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zltrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
rtrace_02�
*__inference_flatten_1_layer_call_fn_524974�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
�
strace_02�
E__inference_flatten_1_layer_call_and_return_conditional_losses_524980�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
ytrace_02�
(__inference_dense_2_layer_call_fn_524989�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
�
ztrace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_525000�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zztrace_0
!:	�2dense_2/kernel
:�2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_525005
*__inference_dropout_1_layer_call_fn_525010�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_525022
E__inference_dropout_1_layer_call_and_return_conditional_losses_525027�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_525036�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_525047�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_1_layer_call_fn_524783conv2d_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_524804conv2d_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_524730conv2d_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_524762conv2d_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
: 2Nadam/Variable
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_72�
#__inference__update_step_xla_524884
#__inference__update_step_xla_524889
#__inference__update_step_xla_524894
#__inference__update_step_xla_524899
#__inference__update_step_xla_524904
#__inference__update_step_xla_524909
#__inference__update_step_xla_524914
#__inference__update_step_xla_524919�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7
�B�
$__inference_signature_wrapper_524879conv2d_2_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jconv2d_2_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_2_layer_call_fn_524928inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524939inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_3_layer_call_fn_524948inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524959inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_max_pooling2d_1_layer_call_fn_524964inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524969inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_1_layer_call_fn_524974inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_1_layer_call_and_return_conditional_losses_524980inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_524989inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_525000inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_1_layer_call_fn_525005inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_1_layer_call_fn_525010inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_525022inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_525027inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_525036inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_525047inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�false_positives
�false_negatives
�intermediate_weights"
_tf_keras_metric
/:-2Nadam/m/conv2d_2/kernel
/:-2Nadam/v/conv2d_2/kernel
!:2Nadam/m/conv2d_2/bias
!:2Nadam/v/conv2d_2/bias
/:-2Nadam/m/conv2d_3/kernel
/:-2Nadam/v/conv2d_3/kernel
!:2Nadam/m/conv2d_3/bias
!:2Nadam/v/conv2d_3/bias
':%	�2Nadam/m/dense_2/kernel
':%	�2Nadam/v/dense_2/kernel
!:�2Nadam/m/dense_2/bias
!:�2Nadam/v/dense_2/bias
':%	�2Nadam/m/dense_3/kernel
':%	�2Nadam/v/dense_3/kernel
 :2Nadam/m/dense_3/bias
 :2Nadam/v/dense_3/bias
�B�
#__inference__update_step_xla_524884gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524889gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524894gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524899gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524904gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524909gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524914gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_524919gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
: (2true_positives
: (2true_negatives
: (2false_positives
: (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2intermediate_weights
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2intermediate_weights
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2intermediate_weights�
#__inference__update_step_xla_524884~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`ധ���?
� "
 �
#__inference__update_step_xla_524889f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_524894~x�u
n�k
!�
gradient
<�9	%�"
�
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_524899f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�����?
� "
 �
#__inference__update_step_xla_524904pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_524909hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_524914pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_524919f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
!__inference__wrapped_model_524631~ !56DE?�<
5�2
0�-
conv2d_2_input���������
� "1�.
,
dense_3!�
dense_3����������
D__inference_conv2d_2_layer_call_and_return_conditional_losses_524939s7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
)__inference_conv2d_2_layer_call_fn_524928h7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
D__inference_conv2d_3_layer_call_and_return_conditional_losses_524959s !7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
)__inference_conv2d_3_layer_call_fn_524948h !7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
C__inference_dense_2_layer_call_and_return_conditional_losses_525000d56/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_2_layer_call_fn_524989Y56/�,
%�"
 �
inputs���������
� ""�
unknown�����������
C__inference_dense_3_layer_call_and_return_conditional_losses_525047dDE0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_3_layer_call_fn_525036YDE0�-
&�#
!�
inputs����������
� "!�
unknown����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_525022e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_525027e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_1_layer_call_fn_525005Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_1_layer_call_fn_525010Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
E__inference_flatten_1_layer_call_and_return_conditional_losses_524980g7�4
-�*
(�%
inputs���������
� ",�)
"�
tensor_0���������
� �
*__inference_flatten_1_layer_call_fn_524974\7�4
-�*
(�%
inputs���������
� "!�
unknown����������
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_524969�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_1_layer_call_fn_524964�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
H__inference_sequential_1_layer_call_and_return_conditional_losses_524730� !56DEG�D
=�:
0�-
conv2d_2_input���������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_524762� !56DEG�D
=�:
0�-
conv2d_2_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
-__inference_sequential_1_layer_call_fn_524783v !56DEG�D
=�:
0�-
conv2d_2_input���������
p

 
� "!�
unknown����������
-__inference_sequential_1_layer_call_fn_524804v !56DEG�D
=�:
0�-
conv2d_2_input���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_524879� !56DEQ�N
� 
G�D
B
conv2d_2_input0�-
conv2d_2_input���������"1�.
,
dense_3!�
dense_3���������