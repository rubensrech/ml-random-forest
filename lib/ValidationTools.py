from math import floor

class ValidationTools:
    @staticmethod
    def bootstrap(D, frac=1):
        train = D.sample(frac=frac, replace=True, random_state=1)
        test = D.drop(train.index)
        return (train, test)

    @staticmethod
    def getFolds(D, targetAttr, K):
        kfolds = {}

        # Calculate number of instances of each class to be in each fold
        classesInstsCount = D.groupby([targetAttr]).agg({ targetAttr: 'count' })
        classesInstsCount['InstsPerFold'] = classesInstsCount.apply(lambda x: x/K)

        classes = D[targetAttr].unique()
        for c in classes:
            # Get instances of class 'c' in 'D'
            classSet = D[D[targetAttr] == c]
            # Get number of instances of class 'c' to be in each fold
            classInstsCount = floor(classesInstsCount.loc[c]['InstsPerFold'])
            # For each fold
            for i in range(K):
                # Sample instances of class 'c'
                classSample = classSet.sample(n=classInstsCount, random_state=1)
                # Add sample to fold
                if i not in kfolds:
                    kfolds[i] = classSample
                else:
                    kfolds[i] = kfolds[i].append(classSample, sort=False)
                # Remove sampled instances from class 'c' instances
                classSet = classSet.drop(classSample.index)
        
        return list(kfolds.values())

    