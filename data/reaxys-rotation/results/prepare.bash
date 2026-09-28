# copy QM9
cp ../../../qm9-rotation/results/sbatch/*.bash .

# remove extra
rm *_power-*.bash *_abs-*.bash *noH*.bash

# rename
for i in *.bash ; do mv $i ${i/QM9Rotation/miniRXOR} ; done

# change wandb name
sed s/cv10-QM9Rotation/cv10-miniRXOR/ miniRXOR*.bash -i

# change dataset
sed -e '/--dataset/d' -e '/train.py/a --dataset data/reaxys-rotation/dataloader_rotation.py:Rotation \\' miniRXOR*.bash -i

# fix patience
sed -i '/patience/d' *.bash

# duplicate
for i in *.bash; do
  cp -iv $i ${i/rot589/rot_exp}
  mv -iv $i ${i/rot589/rot_comp}
done

# target and wandb name
sed -i s/rot589/specific_rotation/ *_exp*.bash
sed -i s/rot589/specific_rotation_computed/ *_comp*.bash

# splitter
sed -i s/qm9-rotation/reaxys-rotation/g *.bash

# add noH
for i in *.bash ; do sed -e '/wandb/a --noH \\' -e 's/-ns/_noH-ns/' $i > ${i/-ns/_noH-ns}; done

for i in $(ls *.bash | grep -v sign); do
  arch=$(echo ${i/.bash/} | sed s/_noH// | cut -d'-' -f3,4,5,6)
  checkpoint=cv/3DMol-rotation-cv/QM9Rotation-rot589/cv10-QM9Rotation-rot589-${arch}-split0.best_checkpoint.pt
  sed -e "/train.py/a --checkpoint ${checkpoint}"' \\' -e s/cv10/cv10ft/ $i > ${i/.bash/.finetune.bash}
done
for i in *sign*.bash ; do
  arch=$(echo ${i/.bash/} | sed s/_noH// | cut -d'-' -f3,4,5,6)
  checkpoint=cv/3DMol-rotation-cv/QM9Rotation-rot589_sign/cv10-QM9Rotation-rot589_sign-${arch}-split0.best_checkpoint.pt
  sed -e "/train.py/a --checkpoint ${checkpoint}"' \\' -e s/cv10/cv10ft/ $i > ${i/.bash/.finetune.bash}
done


sed -i -e '/checkpoint/a --fine_tuning \\' *.finetune.bash
