Frequently Asked Questions
==========================

What is a moka pot?
-----------------------

A moka pot is a stove top coffee maker that typically looks like a small
percolator, although the process of brewing coffee is quite different. Moka pots
are often known as stove top espresso makers: water in the base is heated until
the pressure of the vapor forces liquid water through a packed disk of coffee
grounds. However, moka pots brew at a significantly lower pressure than true
espresso.

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Espressokanne_im_Lichtzelt.jpg/800px-Espressokanne_im_Lichtzelt.jpg
  :width: 200
  :alt: Moka pot image

.. image:: https://upload.wikimedia.org/wikipedia/commons/d/dd/Moka_Animation.gif
  :width: 200
  :alt: Moka pot animation

(Images from `Wikipedia <https://en.wikipedia.org/wiki/Moka_pot>`_)

Why is mokapot named after a moka pot?
--------------------------------------

Percolator was so named in part because the model training algorithm
resembles the process of brewing coffee with a percolator: A few confident PSMs
are used as positive examples for model training, the learned model adds new
confident PSMs to the positive examples, and the process repeats for some number
of iterations. In a percolator, water is heated until it condenses and
drips through the coffee grounds. The longer a percolator brews, the more times
this process is repeated, which results in stronger coffee.

We originally started writing mokapot to re-score PSMs using static models that
were already trained by Percolator or other sources. In a way, we thought this
was nice analogy between how a moka pot compares with percolator for brewing
coffee. Then we decided to implement the full Percolator training procedure as
well, making the analogy break down. Now we're left with a little bit of a
misnomer, but we think that is OK.

What input and output formats does mokapot support?
---------------------------------------------------

mokapot can directly read files that are in the `Percolator tab-delimited file
format
<https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_.
Additionally, any :py:class:`pandas.DataFrame` can be used to define :doc:`a
collection of PSMs <api/dataset>`. This allows mokapot to use any PSMs that you
have parsed into a :py:class:`pandas.DataFrame`, regardless of how they were
stored originally.

Currently, mokapot can output a tab-delimited text table with the results,
similar to the tabular results returned by Percolator. However, the results
stored in the :doc:`confidence objects <api/confidence>` are also
:py:class:`pandas.DataFrame` objects, allowing users to write custom output
files, or conduct further analyses in Python.

Adding support for additional input and output formats is something we're
interested in and working on.


How do I do <insert a task> with mokapot?
-----------------------------------------

The best way to get help with these kinds of questions is to post them to the
`mokapot discussion board <https://github.com/wfondrie/mokapot/discussions>`_.
Posting a question here, instead of asking privately by email, means that all
users can benefit from the answers to your question and others.


Help, I just got a mysterious error! What should I do?
------------------------------------------------------

First, make sure you have read the documentation on this site and can't find an
answer to what went wrong. Also, verify that you have the latest version of
mokapot installed. Finally, check the `mokapot discussion board
<https://github.com/wfondrie/mokapot/discussions>`_ to see if anyone else has
encountered your error.

If you still have a problem, please don't hesitate to `file an issue on Github
<https://github.com/wfondrie/mokapot/issues>`_. It may very well be that you've
found lapses in our documentation or a bug that is frustrating other users too.
When you do open an issue, please make every effort to provide an example that
reproduces the error you've run into. Having a reproducible example makes fixing
a bug much faster and easier.
