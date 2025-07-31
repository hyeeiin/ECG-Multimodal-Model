import { YStack, Text, ScrollView } from 'tamagui'
import QuestionnaireForm from '../../components/QuestionnaireForm'
import ImageUploader from '../../components/ImageUploader'
import SubmitButton from '../../components/SubmitButton'
import { useState, useLayoutEffect } from 'react'
import { useNavigation } from "@react-navigation/native"

export default function FormScreen() {
  const [questionnaire, setQuestionnaire] = useState({})
  const [ecgImage, setEcgImage] = useState(null)
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  const navigation = useNavigation()
    useLayoutEffect(() => {
      navigation.setOptions({
        title: "건강 정보 제출", // ← 원하는 텍스트로 설정
      })
    }, [navigation])

  return (
    <ScrollView>
      <YStack p="$6" space="$4" background="$background">
        <Text fontSize="$6" fontWeight="bold" mb="$2">
          문진 정보 입력
        </Text>

        <QuestionnaireForm data={questionnaire} onChange={setQuestionnaire} />
        <ImageUploader ecgImage={ecgImage} setEcgImage={setEcgImage} />
        <SubmitButton
          questionnaire={questionnaire}
          ecgImage={ecgImage}
          setMessage={setMessage}
          setLoading={setLoading}
          loading={loading}
          message={message}
        />

        {!!message && (
          <Text color="$red10" fontWeight="600" mt="$2"                                                                                                                                                                                                                                                                                                                                         >
            {message}
          </Text>
        )}
      </YStack>
    </ScrollView>
  )
}
