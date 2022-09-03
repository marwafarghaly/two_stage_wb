import torch




class loss_func():

  @staticmethod
  def compute(output, target, global_output=None, perceptual=None):
    if perceptual is None:
      if global_output is None:
        loss = torch.nn.functional.mse_loss(output, target)
      else:
        loss = (torch.nn.functional.mse_loss(output, target) +
                torch.nn.functional.mse_loss(global_output, target))
    else:
      if global_output is None:
        loss = (torch.nn.functional.mse_loss(output, target) +
                0.1 * perceptual(output, target))
      else:
        loss = (torch.nn.functional.mse_loss(output, target) +
                torch.nn.functional.mse_loss(global_output, target) +
                0.1 * perceptual(output, target))

    return loss
  # @staticmethod
  # def compute(output, target, output_global=None):
  #   if output_global is None:
  #     loss = torch.nn.functional.mse_loss(output, target)
  #   else:
  #     loss = (torch.nn.functional.mse_loss(output, target) +
  #             torch.nn.functional.mse_loss(output_global, target))

    return loss