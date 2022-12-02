# Cytocipher - detection of significantly different cell populations in scRNA-seq
![title](https://github.com/BradBalderson/Cytocipher/blob/main/img/cytocipher_icon.png?raw=true)

## For a complete tutorial that installs cytocipher & reproduces the pancreas development analysis, please see [here](https://github.com/BradBalderson/Cytocipher/tree/main/tutorials/cytocipher_pancreas.ipynb).

## Installation 

```
!pip install cytocipher
```
```
import cytocipher as cc
```

## Expected Input
An AnnData object, *data*, that has been processed similarly to the 
[scanpy standard workflow](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
to produce log-cpm normalised data with tentative cluster labels 
(e.g. from Leiden clustering). It's better if the Leiden resolution is high,
 so that there is alot of over-clustering. 
 Cytocipher merges the non-significantly different clusters.

## Code Scoring Minimal Example
Functions below run the marker gene identification, code scoring, & 
subsequent visualisation of the resulting cell by cluster enrichment scores. 

```
cc.tl.get_markers(data, 'leiden')
cc.tl.code_enrich(data, 'leiden')
cc.pl.enrich_heatmap(data, 'leiden')
```
![title](https://github.com/BradBalderson/Cytocipher/blob/main/img/example_heatmap.png?raw=true)

In a jupyter notebook, you can see documentation using, for example:

```
?cc.tl.get_markers
```

## Cluster Merging Minimal Example
Below runs the cluster merging and visualises the heatmap of enrichment 
scores per cell for each of the new merged clusters.

```
cc.tl.merge_clusters(data, 'leiden')
cc.pl.enrich_heatmap(data, 'leiden_merged')
```

To visualise the scores being compared for a given pair of clusters,
the following visualises the scores as violin plots of the enrichment scores
& prints the p-values determined by comparing the scores:

```
cc.pl.sig_cluster_diagnostics(data, 'leiden', plot_pair=('3', '9'))
```
<span style="color:grey">
Input pair ('3', '9')<br />
p=0.9132771265170103 (3 cells; 3 scores) vs (9 cells; 3 scores)<br />
p=0.8128313109661132 (3 cells; 9 scores) vs (9 cells; 9 scores)<br />
</span>

![title](https://github.com/BradBalderson/Cytocipher/blob/main/img/enrichscore_violin_example.png?raw=true)

To get an sense of the upper- and lower- bounds for what is considered
a significant cluster, default parameters plot the violins illustrated above 
for the upper- and lower- bounds of
significant versus non-significant cluster pairs:

```
cc.pl.sig_cluster_diagnostics(data, 'leiden')
```

See the [pancreas tutorial](https://github.com/BradBalderson/Cytocipher/tree/main/tutorials/cytocipher_pancreas.ipynb) 
for more example Cytocipher functionality, including; visual bias checks, 
Sankey diagrams to visualise cluster merging, volcano plots, and more!

## Issues
Please feel free to post an issue on the [github](https://github.com/BradBalderson/Cytocipher/issues) 
if there is a problem, & I'll help you out ASAP.
