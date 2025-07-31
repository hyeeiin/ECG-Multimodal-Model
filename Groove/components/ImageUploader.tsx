import React, { useEffect } from 'react'
import { Button, Image, Text, YStack } from 'tamagui'
import * as ImagePicker from 'expo-image-picker'

export default function ImageUploader({
  ecgImage,
  setEcgImage,
}: {
  ecgImage: any
  setEcgImage: (img: any) => void
}) {
  // ✅ 앱 시작 시 권한 요청
  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync()
      if (status !== 'granted') {
        alert('이미지를 업로드하려면 갤러리 접근 권한이 필요합니다.')
      }
    })()
  }, [])

  // ✅ 이미지 선택
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      quality: 0.8,
    })

    if (!result.canceled && result.assets?.length > 0) {
      const selected = result.assets[0]
      console.log('선택된 이미지:', selected)
      setEcgImage(selected)
    }
  }

  return (
    <YStack space="$3">
      <Text fontWeight="700" fontSize="$5">
        ECG 이미지 (필수)
      </Text>

      <Button onPress={pickImage}>이미지 선택</Button>

      {ecgImage?.uri && (
        <Image
          source={{ width:350, height: 250, uri: ecgImage.uri }}
          width="100%"
          height="100%"
          objectFit="cover"
          borderRadius="$4"
          mt="$2"
          style={{ borderColor: '#d1d5db' , borderWidth: 1, overflow: 'hidden'}}
        />
      )}
      
    </YStack>
  )
}
