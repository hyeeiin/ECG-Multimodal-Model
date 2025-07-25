import React, { useState } from 'react';
import { ScrollView, View, Text } from 'react-native';
import QuestionnaireForm from '../components/QuestionnaireForm';
import ImageUploader from '../components/ImageUploader';
import SubmitButton from '../components/SubmitButton';

export default function EcgUploadScreen() {

  const [questionnaire, setQuestionnaire] = useState<Record<string, string>>({});
  const [ecgImage, setEcgImage] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  return (
    <ScrollView className='bg-white px-4'>
      <Text className="text-xl font-bold mt-6 mb-3">문진 정보 입력</Text>
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
      {message ? (
        <Text className="text-center mt-3 text-red-500 font-semibold">{message}</Text>
      ) : null}
    </ScrollView>
  );
}
