from dataset import * 

BATCH_SIZE = 64 


def main():
    full_dataset = load_full_dataset()
    train_set, val_set, test_set = split_train_val_test_set(
        data_list = full_dataset,
        train_ratio = 0.6,
        val_ratio = 0.2,
        test_ratio = 0.2, 
    )
    
    train_dataloader = DataLoader(
        dataset = train_set,
        batch_size = BATCH_SIZE,
        shuffle = True,
        drop_last = False, 
    )
    val_dataloader = DataLoader(
        dataset = val_set,
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
    )
    test_dataloader = DataLoader(
        dataset = test_set,
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
    )
    
    for i in train_dataloader:
        print(i)
        
        
if __name__ == '__main__':
    main()
