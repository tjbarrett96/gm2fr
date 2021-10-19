for tag in $@;
do
  for file in {BackgroundFit.pdf,results.npy,results.txt,transform.npz,transform.root};
  do
    echo "analysis/results/${tag}/${file}";
  done
done
