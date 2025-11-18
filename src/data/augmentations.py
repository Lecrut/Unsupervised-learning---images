import torch
import torchvision.transforms.functional as TF


class Augmentations:
    """Klasa zawierająca transformacje i augmentacje obrazów"""
    
    @staticmethod
    def rotate_left(image, angle=45):
        """
        Obraca obraz w lewo o zadany kąt
        
        Args:
            image: tensor [C, H, W] lub PIL Image
            angle: kąt obrotu w stopniach (domyślnie 45)
        
        Returns:
            Obrócony obraz
        """
        return TF.rotate(image, angle)
    
    @staticmethod
    def rotate_right(image, angle=45):
        """
        Obraca obraz w prawo o zadany kąt
        
        Args:
            image: tensor [C, H, W] lub PIL Image
            angle: kąt obrotu w stopniach (domyślnie 45)
        
        Returns:
            Obrócony obraz
        """
        return TF.rotate(image, -angle)
    
    @staticmethod
    def random_rotation(image, max_angle=45):
        """
        Losowy obrót obrazu w zakresie [-max_angle, max_angle]
        
        Args:
            image: tensor [C, H, W] lub PIL Image
            max_angle: maksymalny kąt obrotu (domyślnie 45)
        
        Returns:
            Losowo obrócony obraz
        """
        angle = torch.rand(1).item() * 2 * max_angle - max_angle
        return TF.rotate(image, angle)
