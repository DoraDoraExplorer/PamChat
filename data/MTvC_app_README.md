# MTvC_app

##### Description

`MTvC_app`, Multiple Treatments versus Control provides an easy-to-use interface to the statistical analysis of data where multiple categorical Treatment conditions are compared to a single Control condition.

##### Details

* Compares multiple (categorical) Treatment conditions to a single Control condition. E.g. the effect of 3 different compounds on kinase activity relative to the vehicle control. 
* Peptides showing a significant effect between any Treatment conditions are identified using ANOVA.
* Based on the results of the ANOVA, each individual Treatment condition is tested against the Control by Dunnett's test. This test is a more powerful version of the t-test compared to performing multiple separate t-tests.
* The statistical tests assume that the variation within all conditions is equal. For PamChip data after log transformation this is usually a reasonable assumption. Results may be inaccurate if this condition is not met.
* For the Dunnett's test as well as for the approach of applying multiple separate t-test, it has to be taken into account that multiple comparisons are made. As a consequence, the probability of a false positive is actually larger than that suggested by the p-value resulting from the separate test's. In the case of PamChip data there is also multiple testing associated with the multiple peptides in the analysis. Therefore, in this implementation of Dunnett's test there is no multiple testing correction for testing multiple peptides.
* Peptides with similar response to the treatments are clustered.
* This app is implemented in the C1T1T2T3 workflows.

Views:
* Basal Data View: heatmap with peptides sorted according to their mean over all treatments.
* Peptide Cluster View: Line plots of z-score scaled signals per peptide. This view can be used to identify the different patterns of signal as a function of treatment and to find clusters of peptides that have a similar response to treatment. The line plots are colored according to the significant effects between any treatments, resulting from 1-way ANOVA: red indicates significant treatment effect, black means not significant. 
* Ratio Map View: heatmap of log-ratio treatment effects. 
* Ratio Map for Significant Peptides: Ratio Map View filtered for significant peptides.
* Volcano Plot View

##### Output relations

* pvalue.any: p value (from ANOVA) for the probability that there is no difference between any of the Treatments including Control. A low p-value means significant effect between any Treatment-Control pair. 
* pvalue.treatment:	p-value (from Dunnett test) for the probability that there is no difference between the corresponding Treatment and Control. For the Control, this value is NA. A low p-value means significant effect between all Treatment-Control pairs. 
* LFC:	Log2 fold change between Treatment and Control. For the Control, this value is NA.
* Treatment_ClusterIdx (ClusterAnnotation):	cluster indices of peptides. Available as a Spot factor. Peptides with a similar response to Treatment are clustered together.


Operators:

* [ANOVA_Dunett_operator](https://github.com/pamgene/anova_dunnett_operator)
* [mean_operator](https://github.com/tercen/mean_operator)
* [scale_operator](https://github.com/tercen/scale_operator)
* [rank_by_row_mean_operator](https://github.com/pamgene/rank_by_row_mean_operator)
* [model_based_clustering_operator](https://github.com/pamgene/model_based_clustering_operator)


##### See Also

https://pamcloud.pamgene.com/wiki/Wiki.jsp?page=PamApp%20for%20Multiple%20Treatments%20Vs%20Control

