---
layout: leetcode-entry
title: "2785. Sort Vowels in a String"
permalink: "/leetcode/problem/2023-11-13-2785-sort-vowels-in-a-string/"
leetcode_ui: true
entry_slug: "2023-11-13-2785-sort-vowels-in-a-string"
---

[2785. Sort Vowels in a String](https://leetcode.com/problems/sort-vowels-in-a-string/description/) medium
[blog post](https://leetcode.com/problems/sort-vowels-in-a-string/solutions/4281721/kotlin-count-sort/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13112023-2785-sort-vowels-in-a-string?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/df6a61cf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/402

#### Problem TLDR

Sort vowels in a string

#### Intuition

The sorted result will only depend of the vowels frequencies.

#### Approach

Let's use Kotlin API:
* groupBy
* mapValues
* buildString

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```

    fun sortVowels(s: String): String {
      val freq = s.groupBy { it }.mapValues({ it.value.size }).toMutableMap()
      val vl = mutableListOf('A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u')
      val vs = vl.toSet()
      return buildString {
        for (c in s)
          if (c in vs) {
            while (freq[vl.first()].let { it == null || it <= 0 }) vl.removeFirst()
            freq[vl.first()] = freq[vl.first()]!! - 1
            append(vl.first())
          } else append(c)
      }
    }

```

