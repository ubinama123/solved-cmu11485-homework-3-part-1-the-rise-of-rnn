Download Link: https://assignmentchef.com/product/solved-cmu11485-homework-3-part-1-the-rise-of-rnn
<br>
<ul>

 <li>You are required to do this assignment in the Python (version 3) programming language. Do not use any auto-differentiation toolboxes (PyTorch, TensorFlow, Keras, etc) – you are only permitted and recommended to vectorize your computation using the Numpy library.</li>

 <li>We recommend that you look through all of the problems before attempting the first problem. However we do recommend you complete the problems in order, as the difficulty increases, and questions often rely on the completion of previous questions.</li>

 <li>If you haven’t done so, use pdb to debug your code effectively.</li>

</ul>

<h1>Introduction</h1>

In this assignment, you will continue to develop your own version of PyTorch, which is of course called MyTorch (still a brilliant name; a master stroke. Well done!).

<h2>Homework Structure</h2>

Below is a list of files that are <strong>directly relevant </strong>to hw3.

<table width="622">

 <tbody>

  <tr>

   <td width="21"><strong>IM</strong></td>

   <td width="32"><strong>POR</strong></td>

   <td width="50"><strong>TANT:</strong></td>

   <td width="36">First,</td>

   <td width="31">copy</td>

   <td width="23">the</td>

   <td width="27">high</td>

   <td width="42">lighted</td>

   <td width="31">files/</td>

   <td width="22">fold</td>

   <td width="18">ers</td>

   <td width="31">from</td>

   <td width="23">the</td>

   <td width="50">HW3P1</td>

   <td width="31">hand</td>

   <td width="21">out</td>

   <td width="28">over</td>

   <td width="16">to</td>

   <td width="23">the</td>

   <td width="20">cor</td>

   <td width="11">re</td>

   <td width="36">spond</td>

  </tr>

 </tbody>

</table>

–

<table width="245">

 <tbody>

  <tr>

   <td width="20">ing</td>

   <td width="24">fold</td>

   <td width="19">ers</td>

   <td width="29">that</td>

   <td width="25">you</td>

   <td width="30">used</td>

   <td width="15">in</td>

   <td width="28">hw1</td>

   <td width="26">and</td>

   <td width="29">hw2.</td>

  </tr>

 </tbody>

</table>

<strong>NOTE: We recommend you make a backup of your hw3 files before copying everything over, just in case you break code or want to revert back to an earlier version.</strong>

<table width="512">

 <tbody>

  <tr>

   <td width="40"><strong>Next,</strong></td>

   <td width="31">copy</td>

   <td width="26">and</td>

   <td width="35">paste</td>

   <td width="23">the</td>

   <td width="17">fol</td>

   <td width="20">low</td>

   <td width="20">ing</td>

   <td width="31">code</td>

   <td width="35">stubs</td>

   <td width="31">from</td>

   <td width="82">hw3/stubs.py</td>

   <td width="27">into</td>

   <td width="23">the</td>

   <td width="20">cor</td>

   <td width="24">rect</td>

   <td width="28">files.</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Copy Slice(Function) into nn/functional.py.</li>

 <li>Copy Cat(Function) into nn/functional.py.</li>

 <li>Copy unsqueeze(), getitem () and     len () into tensor.py. These are methods for class Tensor.</li>

 <li>Copy function cat into tensor.py. This is an independent function and not a class method.</li>

</ol>

<strong>NOTE: </strong>You may need to define               pow , which implicitly calls the Pow(Function), if you haven’t done so

<table width="486">

 <tbody>

  <tr>

   <td width="11">al</td>

   <td width="37">ready.</td>

   <td width="30">This</td>

   <td width="42">should</td>

   <td width="18">be</td>

   <td width="28">able</td>

   <td width="16">to</td>

   <td width="24">han</td>

   <td width="19">dle</td>

   <td width="20">int</td>

   <td width="15">ex</td>

   <td width="14">po</td>

   <td width="37">nents.</td>

   <td width="40">Check</td>

   <td width="50">HW3P1</td>

   <td width="32">FAQ</td>

   <td width="20">for</td>

   <td width="32">help.</td>

  </tr>

 </tbody>

</table>

<h2>0.1       Running/Submitting Code</h2>

This section covers how to test code locally and how to create the final submission.

<h3>0.1.1         Running Local Autograder</h3>

Run the command below to calculate scores and test your code locally.

./grade.sh 3

If this doesn’t work, converting <a href="https://en.wikipedia.org/wiki/Newline">line-endings</a> may help:

sudo apt install dos2unix dos2unix grade.sh

./grade.sh 3

If all else fails, you can run the autograder manually with this:

python3 ./autograder/hw3_autograder/runner.py

<h3>0.1.2         Running the Sandbox</h3>

We’ve provided sandbox.py: a script to test and easily debug basic operations and autograd.

<strong>Note: We will not provide new sandbox methods for this homework. You are required to write your own from now onwards.</strong>

python3 sandbox.py

<h3>0.1.3         Submitting to Autolab</h3>

<strong>Note: You can submit to Autolab even if you’re not finished yet. You should do this early and often, as it guarantees you a minimum grade and helps avoid last-minute problems with Autolab.</strong>

Run this script to gather the needed files into a handin.tar file:

./create_tarball.sh

If this crashes (with some message about a hw4 folder) use dos2unix on this file too.

You can now upload handin.tar to <a href="https://autolab.andrew.cmu.edu/courses/11485-f20/assessments">Autolab</a> <a href="https://autolab.andrew.cmu.edu/courses/11485-f20/assessments">.</a>

<h1>1           RNN</h1>

In mytorch/nn/rnn.py we will implement a full-fledged Recurrent Neural Network module with the ability to handle variable length inputs in the same batch.

<h2>1.1         RNN Unit</h2>

Follow the starter code available in mytorch/nn/rnn.py to get a better sense of the various attributes of the RNNUnit.

Figure 1: The computation flow for the RNN Unit forward.

)                                                            (1)

The equation you should follow is given in equation 1.

<table width="425">

 <tbody>

  <tr>

   <td width="25">You</td>

   <td width="22">are</td>

   <td width="13">re</td>

   <td width="39">quired</td>

   <td width="16">to</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="31">ment</td>

   <td width="23">the</td>

   <td width="18">for</td>

   <td width="31">ward</td>

   <td width="48">method</td>

   <td width="15">of</td>

   <td width="22">RN</td>

   <td width="38">NUnit</td>

   <td width="28">mod</td>

   <td width="21"> </td>

  </tr>

 </tbody>

</table>

ule. The description of the inputs and

expected outputs are specified below:

<h3>Inputs</h3>

<ul>

 <li>x (effective batch size, input size)

  <ul>

   <li>Input at the current time step.</li>

   <li>NOTE: For now interpret effective batch size same as regular batch size. The difference will become apparent later in the homework. For the definition of effective batch size check Appendix and/or Figure 5</li>

  </ul></li>

 <li>h (effective batch size, hidden size)

  <ul>

   <li>Hidden state generated by the previous time step, <em>h<sub>t</sub></em><sub>−1</sub></li>

  </ul></li>

</ul>

<h3>Outputs</h3>

<ul>

 <li>h prime: (effective batch size, hidden size)</li>

</ul>

<strong>– </strong>New hidden state generated at the current time step, <em>h</em><sup>0</sup><em><sub>t</sub></em>

<strong>NOTE: </strong>As a matter of convention, if you apply certain reshape/transpose operations while using <em>self.weight ih </em>then please do the same for <em>self.weight hh</em>. This is important to note because <em>self.weight hh </em>is symmetric and is therefore exposed to multiple interpretations on how to use it.

<h2>1.2         Detour 1: Cat, Slice, Unsqueeze</h2>

In the following section we implement some necessary functions which will prove to be extremely handy while completing the rest of the homework. These are namely: Cat, Slice and Unsqueeze.

<h3>1.2.1         Cat (7 points)</h3>

Concatenates the given sequence of tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty. Please refer to the PyTorch Documentation for better understanding. If you are not into documentations then please refer back to Recitation 0 where this was covered as well.

First implement the corresponding Cat class in mytorch/nn/functional.py. This will be a subclass of

<table width="87">

 <tbody>

  <tr>

   <td width="29">Func</td>

   <td width="25">tion</td>

   <td width="33">class.</td>

  </tr>

 </tbody>

</table>

<table width="75">

 <tbody>

  <tr>

   <td width="13">of</td>

   <td width="13">se</td>

   <td width="49"> </td>

  </tr>

 </tbody>

</table>

Next implement a helper function <em>cat </em>in the mytorch/tensor.py to call the concatenation operation on a list quences. This should in turn correctly call the corresponding Function sub-class. Below is a description

of the required inputs and outputs.

<h3>Inputs</h3>

<ul>

 <li>seq: list of tensors

  <ul>

   <li>The list basically contains the sequences we want to concatenate</li>

  </ul></li>

 <li>dim: (int, default=0)

  <ul>

   <li>The dimension along which we are supposed to concatenate the tensors in the list seq</li>

  </ul></li>

</ul>

<strong>Outputs </strong>• Tensor:

<ul>

 <li>The concatenated tensor</li>

</ul>

<strong>NOTE: </strong>You don’t need to add anything to the Tensor class in mytorch/tensor.py with respect to Cat operation.

<h3>1.2.2         Slice</h3>

Despite being worth only 5 points, implementing this operation takes your Tensor class to a while new level. With this operation up and running, you can index your Tensor class just like you index Numpy arrays/PyTorch tensors, while autograd takes charge of calculating the required gradients (assuming you implemented the backward correctly).

First implement the corresponding Slice class in mytorch/nn/functional.py. This will be a subclass of

<table width="87">

 <tbody>

  <tr>

   <td colspan="2" width="29">Func</td>

   <td colspan="3" width="25">tion</td>

   <td colspan="4" width="33">class.</td>

   <td colspan="16" width="538"> </td>

   <td width="0"></td>

  </tr>

  <tr>

   <td colspan="3" rowspan="2" width="31">Next</td>

   <td colspan="3" rowspan="2" width="27">add</td>

   <td colspan="2" rowspan="2" width="24">the</td>

   <td colspan="3" rowspan="2" width="70"><em>getitem</em></td>

   <td rowspan="2" width="50">method</td>

   <td rowspan="2" width="17">in</td>

   <td rowspan="2" width="32">your</td>

   <td rowspan="2" width="25">Ten</td>

   <td rowspan="2" width="20">sor</td>

   <td rowspan="2" width="32">class</td>

   <td rowspan="2" width="17">my</td>

   <td rowspan="2" width="107">torch tensor.py,</td>

   <td rowspan="2" width="41">which</td>

   <td rowspan="2" width="17">in</td>

   <td rowspan="2" width="31">turn</td>

   <td rowspan="2" width="31">calls</td>

   <td rowspan="2" width="24">the</td>

   <td rowspan="2" width="30">Slice</td>

   <td width="0"></td>

  </tr>

  <tr>

   <td width="25">func</td>

   <td colspan="3" width="25">tion</td>

   <td colspan="3" width="24">sub</td>

   <td colspan="3" width="33">-class.</td>

   <td colspan="15" width="517"> </td>

   <td width="0"></td>

  </tr>

  <tr>

   <td width="25"></td>

   <td width="3"></td>

   <td width="3"></td>

   <td width="21"></td>

   <td width="4"></td>

   <td width="6"></td>

   <td width="18"></td>

   <td width="9"></td>

   <td width="5"></td>

   <td width="21"></td>

   <td width="35"></td>

   <td width="52"></td>

   <td width="20"></td>

   <td width="35"></td>

   <td width="28"></td>

   <td width="21"></td>

   <td width="36"></td>

   <td width="22"></td>

   <td width="59"></td>

   <td width="47"></td>

   <td width="20"></td>

   <td width="33"></td>

   <td width="34"></td>

   <td width="27"></td>

   <td width="32"></td>

   <td width="0"> </td>

  </tr>

 </tbody>

</table>

<h3>HINT</h3>

The implementation of a slicing operation from scratch may appear to be a daunting task but we will employ a cool trick to get this done in just a few lines of code. Whenever we try to slice a segment of a given tensor using the [] notation, python creates a <em>key </em>(depending on what you provide within []) and passes this to the <em>getitem </em> function. This <em>key </em>is used by the class to provide the appropriate result to the user. The <em>key </em>can be either an integer, a python slice object, a list etc depending on how we slice our object. This is the principle implemented in Numpy for slicing ndarrays. For the purpose of our implementation we will try to leverage this fact to make our task easy. Once we create a <em>getitem </em> method for our tensor class, everytime we slice our tensors the <em>getitem </em> method will be invoked with the appropriate <em>key</em>. Given that the intention of slicing on our tensor is to get the appropriate segment of the underlying ndarray (data attribute) as a tensor object, can you use this <em>key </em>to complete the task.

<h3>1.2.3         Unsqueeze</h3>

Returns a new tensor with a dimension of size one inserted at the specified position. Please refer to the PyTorch Documentation for better understanding. If you are not into documentations then please refer back to Recitation 0 where this was covered as well.

<table width="405">

 <tbody>

  <tr>

   <td width="27">Add</td>

   <td width="23">the</td>

   <td width="66"><em>unsqueeze</em></td>

   <td width="48">method</td>

   <td width="15">in</td>

   <td width="30">your</td>

   <td width="24">Ten</td>

   <td width="19">sor</td>

   <td width="31">class</td>

   <td width="16">my</td>

   <td width="104">torch tensor.py</td>

  </tr>

 </tbody>

</table>

.

<strong>HINT: </strong>Why is this only worth 3 points? Well if you look closely, you might be able to use a function you implemented previously without any extra effort. The name of the function you might need begins with ’R’ in numpy.

<h2>1.3         Detour 2: pack sequence, unpack sequence</h2>

<strong>What’s all the fuss about handling variable length samples in a batch?</strong>

This section is devoted to the construction of objects of class <em>PackedSequence </em>which help us create batches in the context of RNNs.

As you might have guessed by now, handling of variable length sequences in a batch requires way more machinery than what exists in a simple RNN. In this section we will create all the utility functions that will help us with the handling of variable length sequence in a single batch.

Before we proceed let’s understand an important class that packages our packed sequences in a form which makes it easier for us to deal with them. This will also help you better understand why handling variable length sequences in a batch can be a tedious task.

Refer to Appendix for more details.

<table width="622">

 <tbody>

  <tr>

   <td width="39">Please</td>

   <td width="22">use</td>

   <td width="22">the</td>

   <td width="24">Ten</td>

   <td width="19">sor</td>

   <td width="23">Slic</td>

   <td width="20">ing</td>

   <td width="25">and</td>

   <td width="25">Cat</td>

   <td width="16">op</td>

   <td width="14">er</td>

   <td width="33">ations</td>

   <td width="24">you</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="44">mented</td>

   <td width="15">in</td>

   <td width="22">the</td>

   <td width="20">pre</td>

   <td width="11">vi</td>

   <td width="21">ous</td>

   <td width="19">sec</td>

   <td width="30">tions</td>

   <td width="34">while</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="31">ment</td>

  </tr>

 </tbody>

</table>

–

<table width="113">

 <tbody>

  <tr>

   <td width="20">ing</td>

   <td width="34">these</td>

   <td width="27">func</td>

   <td width="32">tions.</td>

  </tr>

 </tbody>

</table>

<h3>1.3.1         pack sequence (15 points)</h3>

<table width="374">

 <tbody>

  <tr>

   <td width="15">In</td>

   <td width="16">my</td>

   <td width="49">torch/n</td>

   <td width="21">n/u</td>

   <td width="44">til.py</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="31">ment</td>

   <td width="23">the</td>

   <td width="48">method</td>

   <td width="93">pack sequence</td>

  </tr>

 </tbody>

</table>

.

This function as the name suggests, packs a list of variable length Tensors into an object of class PackedSequence. A list of tensors is given (as shown in Figure 2) which are to be converted into a PackedSequence object (detailed class description below). This object holds the packed 2d tensor which is later fed into the RNN module for training and testing. Refer to Figures 2, 3 and 4 to better understand what pack sequence does.

Figure 2: List of tensors we want to pack

Figure 3: First we sort the list in a descending order based on number of timesteps in each

Figure 4: Final Packed 2d Tensor

<h3>Class: PackedSequence Attributes</h3>

<ul>

 <li>data

  <ul>

   <li>The actual packed tensor. In the context of this homework this is a 2D tensor. Refer to 4</li>

  </ul></li>

 <li>sorted indices

  <ul>

   <li>1d ndarray (numpy n-dimensional array) of integers containing the indices of elements in the original list when they are sorted based on number of time steps in each sample. Refer to Figure 2 and Figure 3</li>

  </ul></li>

 <li>batch sizes

  <ul>

   <li>1d ndarray of integers where ith element represents the number of samples in the original list having atleast (i+1) timesteps. Refer to Figure 4 for more clarity.</li>

  </ul></li>

</ul>

<h3>Function: pack sequence Inputs</h3>

<ul>

 <li>seq: list of tensors</li>

</ul>

<strong>– </strong>The list contains tensors representing individual instances, each having a variable length.

<h3>Outputs</h3>

<ul>

 <li>PackedSequence:</li>

</ul>

<strong>– </strong>PackedSequence

<h3>1.3.2         unpack sequence</h3>

<table width="388">

 <tbody>

  <tr>

   <td width="15">In</td>

   <td width="16">my</td>

   <td width="49">torch/n</td>

   <td width="21">n/u</td>

   <td width="44">til.py</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="31">ment</td>

   <td width="23">the</td>

   <td width="48">method</td>

   <td width="16">un</td>

   <td width="90">pack sequence</td>

  </tr>

 </tbody>

</table>

.

This function unpacks a given PackedSequence object into a list of tensors that were used to create it. This is also a part of the class PackedSequence.

<strong>Function: unpack sequence Inputs</strong>

<ul>

 <li>ps: PackedSequence</li>

</ul>

<strong>Outputs</strong>

<ul>

 <li>list of Tensors:</li>

</ul>

<h2>1.4         TimeIterator</h2>

<table width="454">

 <tbody>

  <tr>

   <td width="15">In</td>

   <td width="16">my</td>

   <td width="49">torch/n</td>

   <td width="58">n/rnn.py</td>

   <td width="17">im</td>

   <td width="17">ple</td>

   <td width="31">ment</td>

   <td width="23">the</td>

   <td width="18">for</td>

   <td width="31">ward</td>

   <td width="48">method</td>

   <td width="20">for</td>

   <td width="23">the</td>

   <td width="44">TimeIt</td>

   <td width="14">er</td>

   <td width="6">a</td>

   <td width="23">tor</td>

  </tr>

 </tbody>

</table>

.

<table width="398">

 <tbody>

  <tr>

   <td width="15">er</td>

   <td width="26">ates</td>

   <td width="51">through</td>

   <td width="31">time</td>

   <td width="19">by</td>

   <td width="22">pro</td>

   <td width="22">cess</td>

   <td width="20">ing</td>

   <td width="24">the</td>

   <td width="16">en</td>

   <td width="23">tire</td>

   <td width="14">se</td>

   <td width="42">quence</td>

   <td width="16">of</td>

   <td width="58"> </td>

  </tr>

 </tbody>

</table>

For a given input this class it          timesteps. Can be thought to represent a single layer for a given basic unit (interpret this as RNNUnit for now) which is applied at each time step.

<h3>Working of forward method in TimeIterator</h3>

<ul>

 <li>This module’s forward method is tasked with receiving a batch of samples packed in the PackedSequence form.</li>

 <li>The method runs the section of the input corresponding to a single time-step across the batches through an RNNUnit.</li>

 <li>The hidden state returned by the RNNUnit is collected and then the section of the input corresponding to the next time-step across the batches are fed through the RNNUnit along with the previously ( lasttime step) collected hidden state to generate a new hidden-state.</li>

 <li>This process is done iteratively till all time-steps are exhausted for each sample in the batch.</li>

 <li>Follow Figures 5, 6, 7 and 8 to get a better understanding of how TimeIterator processes and given input PackedSequence.</li>

</ul>

The inputs and outputs of the forward method are given below for the

<h3>Class: TimeIterator Forward Method Inputs</h3>

<ul>

 <li>input: PackedSequence</li>

 <li>hidden: Tensor (batch size,hidden size)</li>

</ul>

<h3>Outputs</h3>

<ul>

 <li>PackedSequence: hiddens at each timestep for each sample packaged as a packed sequence</li>

</ul>

<table width="425">

 <tbody>

  <tr>

   <td width="25">Sam</td>

   <td width="25">ples</td>

   <td width="23">are</td>

   <td width="15">or</td>

   <td width="35">dered</td>

   <td width="17">in</td>

   <td width="16">de</td>

   <td width="32">scend</td>

   <td width="20">ing</td>

   <td width="15">or</td>

   <td width="21">der</td>

   <td width="38">based</td>

   <td width="20">on</td>

   <td width="28">num</td>

   <td width="22">ber</td>

   <td width="16">of</td>

   <td width="58"> </td>

  </tr>

 </tbody>

</table>

<ul>

 <li>Tensor (batch size,hidden size): This is the hidden generated by the last time step for each sample joined together. This is a slight deviation from PyTorch.</li>

</ul>

<table width="228">

 <tbody>

  <tr>

   <td width="28">base</td>

   <td width="31">class</td>

   <td width="30">with</td>

   <td width="16">ba</td>

   <td width="45">sic unit</td>

   <td width="15">=</td>

   <td width="22">RN</td>

   <td width="40"> </td>

  </tr>

 </tbody>

</table>

Finally in mytorch/nn/rnn.py create a class <em>RNN </em>as a sub-class of TimeIterator which instantiates the NUnit. Please look at the code for more details.

Figure 5: Iteration 0 for the TimeIterator

Figure 6: Iteration 1 for the TimeIterator

Figure 7: Iteration 2 for the TimeIterator

Figure 8: The final output from the TimeIterator

<h1>2           GRU</h1>

In this section we will explore the world of GRU. We will heavily use the machinery built for RNN to make our lives easier and create a full-fledged working GRU.

<h2>2.1         GRU Unit</h2>

<table width="314">

 <tbody>

  <tr>

   <td width="25">nam</td>

   <td width="20">ing</td>

   <td width="22">con</td>

   <td width="20">ven</td>

   <td width="25">tion</td>

   <td width="31">than</td>

   <td width="23">the</td>

   <td width="18">Py</td>

   <td width="32">torch</td>

   <td width="23">doc</td>

   <td width="7">u</td>

   <td width="24">men</td>

   <td width="12">ta</td>

   <td width="32"> </td>

  </tr>

 </tbody>

</table>

In mytorch/nn/gru.py implement the forward pass for a GRUUnit (though we follow a slightly different tion.) The equations for a GRU cell are the following:

<table width="458">

 <tbody>

  <tr>

   <td width="440"><strong>r</strong><em>t </em>= <em>σ</em>(<strong>W</strong><em>ir</em><strong>x</strong><em>t </em>+ <strong>b</strong><em>ir </em>+ <strong>W</strong><em>hr</em><strong>h</strong><em>t</em>−1 + <strong>b</strong><em>hr</em>)</td>

   <td width="18">(2)</td>

  </tr>

  <tr>

   <td width="440"><strong>z</strong><em>t </em>= <em>σ</em>(<strong>W</strong><em>iz</em><strong>x</strong><em>t </em>+ <strong>b</strong><em>iz </em>+ <strong>W</strong><em>hz</em><strong>h</strong><em>t</em>−1 + <strong>b</strong><em>hz</em>)</td>

   <td width="18">(3)</td>

  </tr>

  <tr>

   <td width="440"><strong>n</strong><em>t </em>= <em>tanh</em>(<strong>W</strong><em>in</em><strong>x</strong><em>t </em>+ <strong>b</strong><em>in </em>+ <strong>r</strong><em>t </em>⊗ (<strong>W</strong><em>hn</em><strong>h</strong><em>t</em>−1 + <strong>b</strong><em>hn</em>))</td>

   <td width="18">(4)</td>

  </tr>

  <tr>

   <td width="440"><strong>h</strong><em>t </em>= (1 −<strong>z</strong><em>t</em>) ⊗<strong>n</strong><em>t </em>+ <strong>z</strong><em>t </em>⊗<strong>h</strong><em>t</em>−1</td>

   <td width="18">(5)</td>

  </tr>

 </tbody>

</table>

Please refer to (and use) the GRUUnit class attributes defined in the init method. You are not expected to define any extra attributes for a working implementation of this class.

The inputs to the GRUCell forward method are <em>x </em>and <em>h </em>represented as <em>x<sub>t </sub></em>and <em>h<sub>t</sub></em><sub>−1 </sub>in the equations above. These are the inputs at time <em>t</em>.

The output of the forward method is <em>h<sub>t </sub></em>in the equations above.

You are required to implement the forward method of this module. The description of the inputs and expected outputs are specified below:

<h3>Inputs</h3>

<ul>

 <li>x (effective batch size, input size)

  <ul>

   <li>Input at the current time step.</li>

   <li>For the definition of effective batch size check Appendix and/or Figure 5</li>

  </ul></li>

 <li>h (effective batch size, hidden size)

  <ul>

   <li>Hidden state generated by the previous time step, <em>h<sub>t</sub></em><sub>−1</sub></li>

  </ul></li>

</ul>

<h3>Outputs</h3>

<ul>

 <li>h prime: (effective batch size, hidden size)</li>

</ul>

<strong>– </strong>New hidden state generated at the current time step, <em>h</em><sup>0</sup><em><sub>t</sub></em>

<h2>2.2         GRU</h2>

<table width="198">

 <tbody>

  <tr>

   <td width="29">class</td>

   <td width="30">with</td>

   <td width="16">ba</td>

   <td width="45">sic unit</td>

   <td width="15">=</td>

   <td width="37">GRUU</td>

   <td width="25"> </td>

  </tr>

 </tbody>

</table>

Finally in mytorch/nn/gru.py create a class <em>GRU </em>as a sub-class of TimeIterator which instantiates the base nit. Please look at the code for more details.

<strong>Appendix</strong>

<h1>A       Glossary</h1>

<ul>

 <li><strong>effective batch</strong>: <em>i<sup>th </sup></em>effective batch refers to the set of <em>i<sup>th </sup></em>timesteps from each sample that are simultaneously fed to the RNNUnit in the (<em>i </em>− 1)<em><sup>th </sup></em>iteration inside the TimeIterator</li>

 <li><strong>effective batch size</strong>: number of samples in an effective batch. Effective batch size of the <em>i<sup>th </sup></em>effective batch is equal to the number of samples containing atleast (i+1) timesteps</li>

</ul>

<strong>B What’s all the fuss about handling variable length samples in one batch?</strong>

<h2>B.1        Importance of batching in Deep Learning</h2>

When training a neural network, we stack together a number of inputs a.k.a batching and pass them together for a single training iteration. From a computational standpoint this helps us make the most of GPUs by parallelizing this computation (done by various frameworks such a PyTorch for you). Moreover ideas like batch normalization fundamentally depend on existence of batches instead of using single samples for training. Therefore there is no denying the importance of batching in the deep learning world.

<h2>B.2          Batching for MLPs and CNNs</h2>

Life in a world with only MLPs and CNNs was simpler from a batching perspective. Each input sample had exactly the same shape. These input samples in the form of tensors (each with the same shape) could then be stacked together along a new dimension to create a higher-dimensional tensor which then becomes our batch. For example consider a simple 1-D input where each sample has 20 features. We now want to create a batch of 256 such samples. Each sample has a shape (20,) which are then stacked along a new dimension (this now becomes dimension 0) to provide a batch of shape (256,20) where the first dimension represents number of samples in a batch while the second dimension represents the number of features.

<h2>B.3        Batching in RNNs</h2>

With RNNs, this becomes very tricky. Each sample in this context has a temporal dimension (atleast in the simplest case). The number of feature at each time step remains fixed for all the samples. Let this be <em>K </em>for the purpose of our discussion. Therefore each sample <em>i </em>has <em>T<sub>i </sub></em>time steps where at each time step we have <em>K </em>features. The <em>i<sup>th </sup></em>sample can then be represented by a tensor of shape (<em>T<sub>i</sub></em>,<em>K</em>). Given that the first dimension is of variable length there is no simple way for us to create batches as we did for MLPs and CNNs. One might argue for fixed time-step inputs, but that severely limits the power of RNNs/GRUs/LSTMs and reduces them to a large fixed MLP. Therefore batching in RNNs require a special setup which we provide via PackedSequences.