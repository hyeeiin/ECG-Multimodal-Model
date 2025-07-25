import React from 'react';
import { View, Button, Image, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function ImageUploader({ ecgImage, setEcgImage }: {
  ecgImage: any,
  setEcgImage: (img: any) => void
}) {

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
    });

    if (!result.canceled) {
      setEcgImage(result.assets[0]);
    }
  };

  return (
    <View className="mb-6">
      <Text className="font-bold mb-2">ECG 이미지 (필수)</Text>
      <Button title="이미지 선택" onPress={pickImage} />
      {ecgImage && (
        <Image
          source={{ uri: ecgImage.uri }}
          className="w-full h-48 mt-3 rounded border border-gray-300"
          resizeMode="contain"
        />
      )}
    </View>
  );
}
