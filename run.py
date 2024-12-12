import os

# train only crop test noise x 2sigmoid mixup(all) test_cleaned epoch 5 -> 7
# os.system('python main.py --data_type 1d --scheduler CosineAnnealingLR --model CNNLSTMWithBatchNorm --optimizer adam --n_class 2 --BATCH_SIZE 64 --LR 0.001 --SR 32000 --N_EPOCHS 100 --phase train --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_63.pth')
# os.system('python main.py --data_type 1d  --phase inference --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_63.pth --model CNNLSTMWithBatchNorm --submit_path ./baseline_submit_mai_59.csv')

# # train only crop test noise x temperature scaling mixup(all) test_cleaned  epoch 1 -> 5 -> 7
# os.system('python main.py --data_type 1d --scheduler ReduceLROnPlateau --model CNNLSTMWithBatchNorm --optimizer adam --n_class 2 --BATCH_SIZE 64 --LR 0.001 --SR 32000 --N_EPOCHS 5 --phase train --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_34.pth')
# os.system('python main.py --data_type 1d  --phase inference --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_34.pth --model CNNLSTMWithBatchNorm --submit_path ./baseline_submit_mai_53.csv')

# train only crop test noise x 2sigmoid mixup(all) test_cleaned epoch 5 -> 7
# os.system('python main.py --data_type 1d --scheduler ReduceLROnPlateau --model CNNLSTMWithBatchNorm --optimizer adam --n_class 2 --BATCH_SIZE 64 --LR 0.001 --SR 32000 --N_EPOCHS 5 --phase train --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_35.pth')
# os.system('python main.py --data_type 1d  --phase inference --model_path ./model_weight_1d_CNNLSTMWithBatchNorm_35.pth --model CNNLSTMWithBatchNorm --submit_path ./baseline_submit_mai_54.csv')

# # RawNet
# os.system('python main.py --data_type 1d --scheduler CosineAnnealingLR --model BiCNNLSTM --optimizer adam --n_class 2 --BATCH_SIZE 32 --LR 0.0001 --SR 32000 --N_EPOCHS 50 --phase train --model_path ./model_weight_1d_CNNTransformerModel_1.pth')
# os.system('python main.py --data_type 1d  --phase inference --model_path ./model_weight_1d_CNNTransformerModel_1.pth --model BiCNNLSTM --submit_path ./baseline_submit_mai_62.csv')

# os.system('python main.py --data_type 1d --scheduler CosineAnnealingLR --model resnet --optimizer adam --n_class 2 --BATCH_SIZE 64 --LR 0.001 --SR 32000 --N_EPOCHS 20 --phase train --model_path ./model_weight_1d_106.pth')
# os.system('python main.py --data_type 1d  --phase inference --model_path ./checkpoint.pt --model resnet --submit_path ./baseline_submit_mai_106.csv')

# train , val sigmoid
os.system('python main.py --data_type 1d --scheduler CosineAnnealingLR --model CNNLSTMWithBatchNorm --optimizer adam --n_class 2 --BATCH_SIZE 32 --LR 0.001 --SR 32000 --N_EPOCHS 1 --phase train --model_path ./model_weight_1d_test.pth')
os.system('python main.py --data_type 1d  --phase inference --model_path ./model_weight_1d_test.pth --model CNNLSTMWithBatchNorm --submit_path ./baseline_submit_mai_test.csv')

