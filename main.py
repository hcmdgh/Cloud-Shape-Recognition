from dataset import * 

BATCH_SIZE = 64 


def main():
    data_arr = prepare_dataset()

    train_set, val_set, test_set = split_train_val_test_set(
        arr = data_arr,
        train_ratio = 0.6,
        val_ratio = 0.2,
        test_ratio = 0.2, 
    )
    
    train_dataloader = DataLoader(
        dataset = ImageDataset(train_set),
        batch_size = BATCH_SIZE,
        shuffle = True,
        drop_last = False, 
    )
    val_dataloader = DataLoader(
        dataset = ImageDataset(val_set),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
    )
    test_dataloader = DataLoader(
        dataset = ImageDataset(test_set),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
    )
    
    for batch_img, batch_label in train_dataloader:
        print(batch_img.shape, batch_label.shape)
        
        
if __name__ == '__main__':
    main()
