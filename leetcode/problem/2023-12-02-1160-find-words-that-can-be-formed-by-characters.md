---
layout: leetcode-entry
title: "1160. Find Words That Can Be Formed by Characters"
permalink: "/leetcode/problem/2023-12-02-1160-find-words-that-can-be-formed-by-characters/"
leetcode_ui: true
entry_slug: "2023-12-02-1160-find-words-that-can-be-formed-by-characters"
---

[1160. Find Words That Can Be Formed by Characters](https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/description/) easy
[blog post](https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/solutions/4352470/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02122023-1160-find-words-that-can?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/EIwFek_6qNM)
![image.png](/assets/leetcode_daily_images/2e69b4a0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/425

#### Problem TLDR

Sum of `words` lengths constructed by `chairs`

#### Intuition

Just use the char frequencies map

#### Approach

Some Kotlin's API:
* groupBy
* sumBy
* all
* let

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1)

#### Code

```kotlin

  fun countCharacters(words: Array<String>, chars: String): Int =
    chars.groupBy { it }.let { freq ->
      words.sumBy {
        val wfreq = it.groupBy { it }
        if (wfreq.keys.all { freq[it] != null
          && wfreq[it]!!.size <= freq[it]!!.size })
        it.length else 0
      }
  }

```

