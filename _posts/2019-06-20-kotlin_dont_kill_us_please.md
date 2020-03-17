---
layout: post
title: Kotlin don't kill us please
---
# Kotlin don't kill us please

Все мы, хотим мы этого или нет, любим котлин.
А вот пользователю вашего приложения любить котлин не обязательно. 
Им гораздо важнее чтобы приложение не падало и выполняло свою основную функцию - работало.
Даже если строка, которую ты хотел показать во всплывающей подсказке пуста. Зачем приложению падать? Мы же фильм смотреть пришли! (мы занимаемся показом видео)

Так вот, если вы уже пишите на котлине, как все успешные люди, то наверное успели заметить такую картину:
```
kotlin.UninitializedPropertyAccessException: lateinit property mChatAdapter has not been initialized // что-то вроде вот этого 
```
Все мы конечно любим абсолютно правильный безошибочный код, но пора признаться себе, такого кода не существует (пока)

Постойте, у меня есть решение. Давайте наше приложение будет делать максимально возможное, чтобы выполнить основную функцию. А всякие проверки, ассерты и прочее - пусть это куда-нибудь логгируется, отправляется, пишется. 
Вообще, мы так и писали наше приложение. Если есть где-то ошибка, надо быстро ее поймать, отправить лог в fabric, но приложение не крашить.

Хорошо, но как быть с котлином?

Тут нам поможет Proguard. Мы уже умеем вырезать логи, убирать лишние конкатенации строк https://www.guardsquare.com/en/products/proguard/manual/examples#logging

Теперь наш враг (в релизе, конечно, в дебаге и процессе разработки - помощник) вот этот класс https://github.com/JetBrains/kotlin/blob/v1.3.40/libraries/stdlib/jvm/runtime/kotlin/jvm/internal/Intrinsics.java

Добавим в прогвард его!


```
-assumenosideeffects class kotlin.jvm.internal.Intrinsics {

	public static void checkNotNull(...);

	public static void throwNpe(...);

	public static void throwUninitializedProperty(...);

	public static void throwUninitializedPropertyAccessException(...);

	public static void throwAssert(...);

	public static void throwIllegalArgument(...);

	public static void throwIllegalArgument(...);

	public static void throwIllegalState(...);

	public static void throwIllegalState(...);

	public static void checkExpressionValueIsNotNull(...);

	public static void checkNotNullExpressionValue(...);

	public static void checkReturnedValueIsNotNull(...);

	public static void checkReturnedValueIsNotNull(...);

	public static void checkFieldIsNotNull(...);

	public static void checkParameterIsNotNull(...);

	public static void checkNotNullParameter(...);

	private static void throwParameterIsNullException(...);

	public static void throwUndefinedForReified(...);

	public static void throwUndefinedForReified(...);

	public static void reifiedOperationMarker(...);

	public static void needClassReification(...);

	public static void checkHasClass(...);

}
```

Enjoy 100% crash free!
