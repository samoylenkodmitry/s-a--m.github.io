---
layout: leetcode-entry
title: "2024. Maximize the Confusion of an Exam"
permalink: "/leetcode/problem/2023-07-07-2024-maximize-the-confusion-of-an-exam/"
leetcode_ui: true
entry_slug: "2023-07-07-2024-maximize-the-confusion-of-an-exam"
---

[2024. Maximize the Confusion of an Exam](https://leetcode.com/problems/maximize-the-confusion-of-an-exam/description/) medium
[blog post](https://leetcode.com/problems/maximize-the-confusion-of-an-exam/solutions/3730076/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/7072023-2024-maximize-the-confusion?sd=pf)
![image.png](/assets/leetcode_daily_images/67bb9113.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/268
#### Problem TLDR
Max same letter subarray replacing `k` letters
#### Intuition
An important example is `ftftftft k=3`: we must fill all the intervals. It also tells, after each filling up we must decrease `k`. Let's count `T` and `F`.
Sliding window is valid when `tt <= k || ff <= k`.
#### Approach
We can save some lines using Kotlin collections API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or $$O(1)$$ using `asSequence`

#### Code

```kotlin

fun maxConsecutiveAnswers(answerKey: String, k: Int): Int {
    var tt = 0
    var ff = 0
    var lo = 0
    return answerKey.mapIndexed { i, c ->
        if (c == 'T') tt++ else ff++
        while (tt > k && ff > k && lo < i)
        if (answerKey[lo++] == 'T') tt-- else ff--
        i - lo + 1
    }.max() ?: 0
}

```

