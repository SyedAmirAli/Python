for (let i = 1; i <= 6; i++) {
  console.log(
    `predicted_class_${i}, confidence_${i} = predict_class(image_path_${i})\nprint(f'Predicted Class: {predicted_class_${i}} with confidence: {confidence_${i} * 100:.2f}%/n')
    `
  );
}
