

def make_model(sess, model_to_run, model_path, labels_path):
  """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  if model_to_run == 'InceptionV3':
    mymodel = model.InceptionV3Wrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  else:
    raise ValueError('Invalid model name')

  return mymodel

sess = utils.create_session()

mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)