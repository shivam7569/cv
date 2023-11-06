def inception_loss(outputs, labels, primitive_loss_fn):
    aux_classifier_1_out, aux_classifier_2_out, main_classifier_out = outputs
    
    aux_classifier_1_loss = primitive_loss_fn(aux_classifier_1_out, labels)
    aux_classifier_2_loss = primitive_loss_fn(aux_classifier_2_out, labels)
    main_classifier_loss = primitive_loss_fn(main_classifier_out, labels)

    loss = main_classifier_loss + 0.3 * (aux_classifier_1_loss + aux_classifier_2_loss)

    return loss

def inceptionv2_loss(outputs, labels, primitive_loss_fn):
    aux_classifier_out, main_classifier_out = outputs
    
    aux_classifier_loss = primitive_loss_fn(aux_classifier_out, labels)
    main_classifier_loss = primitive_loss_fn(main_classifier_out, labels)

    loss = main_classifier_loss + (0.3 * aux_classifier_loss)

    return loss