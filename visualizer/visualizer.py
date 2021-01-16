from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud
from matplotlib.colors import LogNorm
from PIL import Image
from sklearn.metrics import (auc, confusion_matrix,
                             multilabel_confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelBinarizer


class RedditVisualizer:
    """
    Functions to vizualize text data

    Parameters:
    X: pd.Series, column or corpus to visualize.  Cannot be a sparse matrix.
    y: labels
        NOTE: y must not be transformed
    transformer: Transformer to use for visualizations
    classifier: Classifier to use for model metrics

    Example:
    tfidf = TfidfVectorizer(max_features=5000)
    logreg = LogisticRegression(C=5)
    viz = Visualizer(X_transformed, y, transformer=tfidf, classifier=logreg)

    Example using transformer and model from a trained pipeline:



    """

    def __init__(self, X, y, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
        self.X = X
        self.y = y
        self.labels_ = np.unique(y)
        self.pairs = list(combinations(self.labels_, 2))
        self.features_df_ = None

    def check_create_features_df(self):
        '''Creates the features dataframe used for comparison between classes if it does not exist

            Returns:
            features_df_
        '''
        if self.features_df_ is None:
            features_data = self.transformer.transform(self.X).toarray()
            # features_data = self.X.toarray()
            features_columns = self.transformer.get_feature_names()

            self.features_df_ = pd.DataFrame(data=features_data,
                                             columns=features_columns)

    def make_cloud(self, height=300, width=800, max_words=100,
                   split=None, stopwords=None, colormap='viridis', background_color='black'):
        '''
        Inputs:
        text_column: name of text column in dataframe
        labels_column: column that contains the labels, if split=True
        height: height of each wordcloud
        width: width of each wordcloud
        max_words: max words for each wordcloud
        split: if True, wordcloud for each subreddit
        labels: must provide list of labels if split=True, to generate a wordcloud for each label
        stopwords: usually these are the same stopwords used by the tranformer (CountVectorizer or Tfidf)
        colormap: any choice from matplotlib gallery.  Find them with plt.cm.datad
            'random': picks a random colormap
        '''

        colormaps = [m for m in plt.cm.datad if not m.endswith("_r")]
        wc = wordcloud.WordCloud(max_words=max_words,
                                 width=width,
                                 height=height,
                                 background_color=background_color,
                                 colormap=np.random.choice(
                                     colormaps) if colormap == 'random' else colormap,
                                 stopwords=stopwords)
        if split:
            for label in self.labels_:
                label_mask = self.y == label
                cloud = wc.generate(self.X[label_mask].str.cat())
                plt.figure(figsize=(width / 100, height * len(self.labels_) / 100), dpi=100)
                plt.title(label.upper(), fontdict={'fontsize': 15})
                plt.axis("off")
                plt.imshow(cloud.to_image(), interpolation='bilinear')

        else:
            cloud = wc.generate(self.X.str.cat())
            return cloud.to_image()

    def cloud_image_mask(self,
                         image,
                         figsize=(12, 12),
                         max_words=500,
                         colormap='Reds',
                         outline_color='orangered',
                         stopwords=None,
                         background_color='white',
                         reverse=False):
        '''Creates wordcloud using an image as a mask, but uses the image colors for the wordcloud.

        Parameters:
        image: image to use for mask
        colormap: colormap to use for words, from matplotlib choices
        outline_color: color for outline of the image
        reverse: switch whether the words are inside or outside of the masked image
        '''
        img = Image.open(image)
        gray = np.array(img.convert('L'))
        mask = np.where(gray < 200, 255, 0)
        if reverse:
            mask = np.where(gray > 200, 255, 0)

        wc = wordcloud.WordCloud(background_color=background_color,
                                 max_words=max_words,
                                 mask=mask,
                                 colormap=colormap,
                                 contour_color=outline_color,
                                 contour_width=1,
                                 stopwords=stopwords)
        wc.generate(self.X.str.cat())
        plt.figure(figsize=figsize)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis(False)
        plt.show()

    def cloud_colored_image_mask(self,
                                 image,
                                 figsize=(12, 12),
                                 max_words=1000,
                                 stopwords=None,
                                 background_color='white',
                                 include_original=False):
        '''Creates wordcloud using an image as a mask, but uses the image colors for the wordcloud.

        Parameters:
        image: image to use for mask
        include_original: display the original image alongside the word cloud
            (be aware that they will be the same image when drawn)
        '''
        mask = np.array(Image.open(image))
        colorcloud = wordcloud.WordCloud(stopwords=stopwords,
                                         background_color=background_color,
                                         mode="RGBA",
                                         max_words=max_words,
                                         mask=mask)
        colorcloud.generate(self.X.str.cat())
        image_colors = wordcloud.ImageColorGenerator(mask)
        if include_original:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            ax1.imshow(mask)
            ax1.axis(False)
            ax2.imshow(colorcloud.recolor(color_func=image_colors),
                       interpolation="bilinear")
            ax2.axis(False)
            plt.show()
        else:
            plt.figure(figsize=figsize)
            plt.imshow(colorcloud.recolor(color_func=image_colors),
                       interpolation="bilinear")
            plt.axis(False)
            plt.show()

    def plot_most_common_features(self, num_features=20, standardize=False, include_combined=False):
        '''
        Plots the most common features for each subreddit in the DataFrame

        Parameters:
        num_features: number of most common features to plot for each subreddit

        standardize: put all of the plots on the same scale

        combined: include a plot of the most common features of all of the subreddits combined

        Returns:

        multiple plots

        '''
        self.check_create_features_df()

        fig, ax = plt.subplots(ncols=1,
                               nrows=len(self.labels_) + int(1 if include_combined else 0),
                               figsize=(15, num_features / 1.3 * len(self.labels_)))

        for subplot_idx, label in enumerate(self.labels_):
            label_mask = (self.y == label).values
            label_features = self.features_df_.loc[label_mask]
            label_top_words = label_features.sum().sort_values(
                ascending=False).head(num_features)[::-1]
            label_top_words.plot(kind='barh', ax=ax[subplot_idx])
            ax[subplot_idx].set_title(
                f'{num_features} Most Common Words for {label.upper()}', fontsize=16)

            if standardize:
                max_occurence = self.features_df_.sum().max() * 1.02
                ax[subplot_idx].set_xlim(0, max_occurence)

        if include_combined:
            most_common = self.features_df_.sum().sort_values(
                ascending=False).head(num_features)[::-1]
            most_common.plot(kind='barh', ax=ax[subplot_idx + 1])
            ax[subplot_idx + 1].set_title(
                f'{num_features} Most Common Words for ({", ".join(self.labels_).upper()})')

            if standardize:
                ax[subplot_idx + 1].set_xlim(0, max_occurence)

        plt.tight_layout(h_pad=7)

    def plot_most_common_pairs(self, num_features=20):
        '''
        Plots the most common features for each subreddit in the DataFrame

        Parameters:

        num_features: number of most common features to plot for each subreddit

        Returns:

        plots

        '''
        self.check_create_features_df()

        fig, ax = plt.subplots(ncols=2,
                               nrows=len(self.pairs),
                               figsize=(16, (num_features / 1.5) * len(self.pairs)))

        for i, pair in enumerate(self.pairs):

            # features for each pair

            mask_0 = (self.y == pair[0]).values
            mask_1 = (self.y == pair[1]).values

            feats_0 = self.features_df_.loc[mask_0]
            feats_1 = self.features_df_.loc[mask_1]
            # combined
            common_feats = feats_0.append(feats_1)
            # this is the most common between the two
            most_common = common_feats.sum().sort_values(
                ascending=False).head(num_features)[::-1]
            # plot
            feats_0[most_common.index].sum().plot.barh(
                ax=ax[i, 0], color='navy')
            feats_1[most_common.index].sum().plot.barh(
                ax=ax[i, 1], color='orange')
            ax[i, 0].set_title(
                f'Top {num_features} - {pair} \nSub: {pair[0].upper()}', fontsize=16, wrap=True)
            ax[i, 1].set_title(
                f'Top {num_features} - {pair} \nSub: {pair[1].upper()}', fontsize=16, wrap=True)
            max_occurence = common_feats.sum().max() * 1.02
            ax[i, 0].set_xlim(0, max_occurence)
            ax[i, 1].set_xlim(0, max_occurence)
        plt.tight_layout()

    def plot_most_common_bar(self, num_features=20, stacked=False):

        self.check_create_features_df()

        most_common = self.features_df_.sum().sort_values(
            ascending=False).head(num_features)
        groups = self.features_df_.groupby(self.y).sum()[
            most_common.index].T.head(num_features)

        fig, ax = plt.subplots(figsize=(20, 10))

        if stacked is False:
            groups.plot.bar(ax=ax, width=.8, fontsize=15)
        else:
            groups.plot(kind='bar', ax=ax, width=.35,
                        fontsize=15, stacked=True, )

        ax.set_title(f'{num_features} Most Common Words', fontsize=20)
        ax.set_ylabel('# of Occurences', fontsize=15)
        ax.legend(fontsize=15, fancybox=True,
                  framealpha=1, shadow=True, borderpad=1)

    def plot_coef_feature_importance(self, figsize=(16, 12), n_features=10, colormap='Blues_d'):
        '''Plots the feature importance seaparately for each class

        Parameters:
        model: trained model to analyze
        n_features: number of top and bottom features to display
        colormap: choose from matplotlib colormaps
        '''
        if hasattr(self.model, 'coef_'):
            coef_dict = {}
            for i, label in enumerate(self.labels_):
                coef_dict[label] = pd.DataFrame(
                    data=self.model.coef_[i],
                    index=self.transformer.get_feature_names())
        else:
            raise AttributeError('Model does not have coefficients')
        for label, coef_df in coef_dict.items():

            plt.figure(figsize=(16, 12))
            plt.style.use('seaborn-poster')

            top_10 = coef_df.sort_values(0, ascending=False)[0].head(n_features)
            bottom_10 = coef_df.sort_values(0, ascending=False)[0].tail(n_features)
            top_and_bottom = pd.DataFrame(data=top_10.append(bottom_10))
            sns.barplot(x=top_and_bottom[0], y=top_and_bottom.index, palette=colormap)

            plt.title(f'Feature Importance for {label.upper()}', fontsize=20)
            plt.xlabel('Coefficients', fontsize=18)


class ClassificationResultsVisualizer:
    """Visualize the classification results of a trained model.

    This requires y_pred and y_proba to already be defined.
    Design choice
    """

    def __init__(self, transformer, model, y_true=None, y_pred=None, y_proba=None):
        self.transformer = transformer
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.labels_ = np.unique(y_true)

    def plot_roc_curve(self, fpr, tpr, label):
        '''ROC curve plot helper'''
        area = auc(fpr, tpr)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='orange', label=f'{str.upper(label)} AUC - {round(area, 4)}')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve for {str.upper(label)}')
        plt.legend(loc='lower right')

    def plot_class_roc_curves(self, y_true, y_proba):
        '''Plots an ROC curve for each class'''
        bin = LabelBinarizer()
        y_true_bin = bin.fit_transform(y_true)
        for i, label in enumerate(self.labels_):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_proba[:, i])
            self.plot_roc_curve(fpr, tpr, label)

    def display_probabilities_df(self):
        '''Display the probabilities along with predicted and actual classes as df'''
        probs_df = pd.DataFrame(data=self.y_proba, columns=self.labels_)
        probs_df['predicted'] = self.y_pred
        probs_df['actual'] = self.y_true
        return probs_df

    def plot_probability_distribution_overlaid(self, y_proba, bins=100):
        '''Plots overlaid probability distribution for each class'''
        for i, label in enumerate(self.labels_):
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins, label=label)
            plt.hist(y_proba[:, i], **kwargs)
            plt.legend()

    def plot_class_probability_distribution(self, y_proba, figsize=(10, 7), bins=100, match_scale=False):
        '''Plot probability distribution'''
        # find the max height of all the histograms to use for ylim
        y_max = max([np.histogram(y_proba[:, i], bins=bins)[0].max() for i, _ in enumerate(self.labels_)]) * 1.05

        for i, label in enumerate(self.labels_):
            with plt.style.context('bmh'):
                plt.figure(figsize=figsize)
                plt.hist(y_proba[:, i], bins=bins, histtype='stepfilled', alpha=0.3)
                plt.title(f'Distribution of P(Outcome = 1) {str.upper(label)}', fontsize=22)
                plt.xlim(-0.1, 1.1)
                if match_scale:
                    plt.ylim(0, y_max)
                plt.ylabel('Frequency', fontsize=18)
                plt.xlabel('Predicted Probability', fontsize=18)

    def plot_confusion_matrix(self, y_true, y_pred, cmap='Blues'):
        '''
        Plots confusion matrix for fitted model, better than scikit-learn version
        '''
        cm = confusion_matrix(y_true, y_pred)
        fontdict = {'fontsize': 16}
        fig, ax = plt.subplots(figsize=(2.2 * len(self.labels_), 2.2 * len(self.labels_)))

        sns.heatmap(cm,
                    annot=True,
                    annot_kws=fontdict,
                    fmt="d",
                    square=True,
                    cbar=False,
                    cmap=cmap,
                    ax=ax,
                    norm=LogNorm(),  # to get color diff on small values
                    vmin=0.00001  # to avoid non-positive error for '0' cells
                    )

        ax.set_xlabel('Predicted labels', fontdict=fontdict)
        ax.set_ylabel('True labels', fontdict=fontdict)
        ax.set_yticklabels(
            labels=self.labels_, rotation='horizontal', fontdict=fontdict)
        ax.set_xticklabels(labels=self.labels_, rotation=20, fontdict=fontdict)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

    def plot_confusion_matrices(self, y_true, y_pred, colormap='Purples'):
        '''Plots individual confusion matrix for each class label'''
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        # May need these for ROC curve
        # mtn = mcm[:, 0, 0]
        # mtp = mcm[:, 1, 1]
        # mfn = mcm[:, 1, 0]
        # mfp = mcm[:, 0, 1]

        fig, ax = plt.subplots(ncols=2, nrows=len(self.labels_),
                               figsize=(15, 6 * len(self.labels_)))
        fontdict = {'fontsize': 16}

        for i, cm in enumerate(mcm):
            sns.heatmap(cm,
                        annot=True,
                        annot_kws=fontdict,
                        fmt="d",
                        square=True,
                        cbar=False,
                        cmap=colormap,
                        ax=ax[i, 0],
                        norm=LogNorm(),  # to get color diff on small values
                        vmin=0.00001  # to avoid non-positive error for '0' cells
                        )
            ax[i, 0].set_xlabel('Predicted', fontdict=fontdict)
            ax[i, 0].set_ylabel('Actual', fontdict=fontdict)
            ax[i, 0].set_yticklabels(labels=['F', 'T'], rotation='horizontal')
            ax[i, 0].set_xticklabels(labels=['F', 'T'])
            ax[i, 0].xaxis.tick_top()
            ax[i, 0].xaxis.set_label_position('top')
            ax[i, 0].set_title(self.labels_[i].upper())

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            recall = tp / (tp + fn)
            fscore = (2 * recall * precision) / (recall + precision)

            box_text = f'''
                        Subreddit: {self.labels_[i].upper()}\n
                        Type 1 Errors (FP): {round(fp,4)}\n
                        Type 2 Errors (FN): {round(fn,4)}\n
                        Precision: {round(precision, 4)}\n
                        Recall / Sensitivity: {round(recall,4)}\n
                        Specificity: {round(specificity,4)}\n
                        F-Score: {round(fscore,4)}
                        '''
            ax[i, 1].text(0.1, 0.5,
                          box_text,
                          bbox=dict(
                              boxstyle="round",
                              ec=('k'),
                              lw=3,
                              fc=(.9, .9, .9),
                          ),
                          horizontalalignment='left',
                          verticalalignment='center',
                          fontsize=20)
            ax[i, 1].set_axis_off()

        plt.tight_layout()

    def plot_probability_distribution_pairs(self, y_proba, figsize=(15, 7), bins=25):
        '''Plots probability distribution for each pair of labels'''
        number_pairs = list(combinations(np.arange(len(self.labels_)), r=2))
        pairs = list(combinations(self.labels_, r=2))
        for pair, number in zip(pairs, number_pairs):
            plt.figure(figsize=figsize)
            hst0 = plt.hist(y_proba[:, number[0]],
                            bins=bins,
                            alpha=0.6,
                            label=pair[0])
            hst1 = plt.hist(1 - y_proba[:, number[1]],
                            bins=bins,
                            alpha=0.6,
                            label=pair[1])
            plt.vlines(x=0.5,
                       ymin=0,
                       ymax=max(hst1[0].max(), hst0[0].max()),
                       color='k',
                       linestyle='--')

            plt.title(f'Probability Distribution across {pair[0]} and {pair[1]}', fontsize=22)
            plt.ylabel('Frequency', fontsize=18)
            plt.xlabel('Predicted Probability', fontsize=18)
            plt.legend(fontsize=20)
