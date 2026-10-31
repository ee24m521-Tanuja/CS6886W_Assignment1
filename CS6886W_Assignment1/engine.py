def run_epoch(model,loader,criterion,device,optimizer=None):
    training=optimizer is not None; model.train(training); total=correct=0; loss_sum=0.
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        out=model(x); loss=criterion(out,y)
        if training: loss.backward(); optimizer.step()
        total+=y.size(0); correct+=out.argmax(1).eq(y).sum().item(); loss_sum+=loss.item()*y.size(0)
    return loss_sum/total,100.*correct/total
