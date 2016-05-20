import sklearn


def main():
    data = construct_data_matrix()
    labels = construct_labels_matrix()

    sklearn.fit(data, labels)


if __name__ == '__main__':
    main()

