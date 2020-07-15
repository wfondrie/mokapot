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
------------------------------------------

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
