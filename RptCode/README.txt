The RptCode directory contains Python scripts developed to work with
Repeat Masker, create new objects that are easier to deal with, and
aid in working with those object.  Files in this directory include:

RMfileReader.py: This is for reading in the contentent of RepeatMasker
output.
* It requires two files: <seq>.fa.out and <seq>.fa.align.  These are
RepeatMasker output that fully describes each instance of a modern
sequence.
* You load this with the createRepObjs command:
     createRepObjs("chr22.fa.align", "chr22.fa.out", "rpt_exclude.txt")
(The last file is a list of repeats that should not be used in any
analysis.)  This will return an array of Repeat objects.  Each object
corresponds to one repeat, and will fully describe the repeat.  In
Python, import the module and type help(Repeat) for a list of
attributes.
* The list of objects can be pickled -- saved to a file.  Its faster
to load them from a pickled file the the .fa/.fa.out files. 
  -- By convention, we use the .X.prm extion to name these pickled
     files, where X indicated if we were using python 2 or 3.  (The
     pickeling is not compatable.)
  -- Using Python 3, we would pickle like:
       pickleRepeats(rpt_list, "chr22.3.prm")
  -- You can now reload with:
       G = list(unpickleRepeat("chr22")) # The ".3.prm" will be
       automatically added.

Finally: you can use the file to automatically create prm files at
once.  From the command line, the command:

   python RMfileRead.py

will got though every file of the form chr*.fa.out in the directoy,
and create the file chr*.3.prm.  (Or .2.prm, ad appopraite.)  This can
take a while.

(FYI: "prm" = "Pickled Repeat Masker"



RptMatrix.py: This is a class for representing repeat objects as
substitution matrices.  We sotre these in psm files.  ("Pickled
Substitution Matrix")
* We can create these from prm files:
    R = unpickleRepeats("chr22.3.prm")
    S = prmList2psm(R)
* Each object is a RptMatrix object.  (Use help so the attributes.)
* Eaach object can also be treated as a matrix.  If O is a RptMatrix
  object, the O[1][2] tells us the number of C->G substitutions.
  (0=A, 1=C, 2=G, 3=T -- alphabetical numbering of bases).
There are easy eays to load in only those repeats that lie within a
gene -- we will owrry about this later.


asymm_tools.py: toold for checking for asymmetric substitution.  We
will worry about it later.

fp_check.py: Functions for floating-point comparison that allow for a
threshold to account for rounding error.

