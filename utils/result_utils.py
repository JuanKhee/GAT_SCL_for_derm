import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_val_curve(train_df, val_df):
    df = train_df.merge(val_df, on='epoch', suffixes=['_train','_val'], )
    fig, ax = plt.subplots(1,2)

    sns.lineplot(df, x="epoch", y="loss_train", ax=ax[0])
    sns.lineplot(df, x="epoch", y="loss_val", ax=ax[0])
    ax[0].legend(labels=['training_loss', 'testing_loss'])
    sns.lineplot(df, x="epoch", y="acc_train", ax=ax[1])
    sns.lineplot(df, x="epoch", y="acc_val", ax=ax[1])
    ax[1].legend(labels=['validation_accuracy', 'validation_accuracy'])
    plt.show()

    return df

if __name__ == "__main__":

    train_results_df = pd.read_csv(
        r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\vgg16_results\training_loss.csv'
    )
    test_results_df = pd.read_csv(
        r'C:\Users\HP-VICTUS\PycharmProjects\pythonProject\GAT_SCL_for_derm\final_results\vgg16_results\test_loss.csv')

    plot_train_val_curve(train_results_df, test_results_df)