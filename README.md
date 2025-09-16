# CS2023Data


## OBJECTIVE

This is a package of scripts and data for analysing a text based on the Body of Knowledge (BOK) of CS2023 [CS2023]. Outputs consist of the rates in 17 Knowledge Area (KA)s of CS2023.

## FILE

* analyze_documents_with_CS2023.py
The script to analyse a text. This uses lda-c [LDA] internally.
* final.model.text
The model obtained by analyzing CS2023. This was extracted by ssLDA [ssLDA] and is used by the lda-c.
* settings.conf  
The parameters used by LDA.
* word.csv  
3304 words in CS2023 BOK.

## PROCEDURE

1. Set up the Python environment.
2. Install lda-c from https://github.com/blei-lab/lda-c/
3. Place the following files in a working directory.  
`analyze_documents_with_CS2023.py, word.csv, settings.conf, final.model.text`
4. Place target files in `target_dir`.
5. Run the following command in the directory.  
`./analyze_with_CS2013.pl --target target_dir`
6. Results are in `{target_dir}result.csv`.  

## REFERENCES

[CS2023] Amruth N. Kumar, Rajendra K. Raj, Sherif G. Aly, Monica D. Anderson, Brett A. Becker, Richard L. Blumenthal, Eric Eaton, Susan L. Epstein, Michael Goldweber, Pankaj Jalote, Douglas Lea, Michael Oudshoorn, Marcelo Pias, Susan Reiser, Christian Servin, Rahul Simha, Titus Winters, and Qiao Xiang. Computer Science Curricula 2023. Association for Computing Machinery, New York, NY, USA, https://doi.org/10.1145/3664191, 2024.

[LDA] David M. Blei, LDA-C, https://github.com/blei-lab/lda-c/ (accessed 2025-09-16).

[ssLDA] T. Sekiya, Y. Matsuda, and K. Yamaguchi. Curriculum Analysis of CS Departments Based on CS2013 by Simplified, Supervised LDA. LAK’15, Proceedings of the Fifth International Conference on Learning Analytics And Knowledge, pp. 330-339, NY, USA, 2015.


## LICENSE

This software (CS2023Data) is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

CS2023Data is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.

Takayuki Sekiya
sekiya[at]ecc.u-tokyo.ac.jp

(C) Copyright 2025, Takayuki Sekiya (sekiya[at]ecc.u-tokyo.ac.jp)
