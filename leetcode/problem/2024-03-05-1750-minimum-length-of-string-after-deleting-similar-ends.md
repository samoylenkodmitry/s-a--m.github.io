---
layout: leetcode-entry
title: "1750. Minimum Length of String After Deleting Similar Ends"
permalink: "/leetcode/problem/2024-03-05-1750-minimum-length-of-string-after-deleting-similar-ends/"
leetcode_ui: true
entry_slug: "2024-03-05-1750-minimum-length-of-string-after-deleting-similar-ends"
---

[1750. Minimum Length of String After Deleting Similar Ends](https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/description/) medium
[blog post](https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/solutions/4825399/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05032024-1750-minimum-length-of-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xuqYQW-zyMo)
![image.png](/assets/leetcode_daily_images/c50bc1e8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/529

#### Problem TLDR

Min length after trimming matching prefix-suffix several times. #medium

#### Intuition

By looking at the examples, greedy approach should be the optimal one.

#### Approach

* careful with indices, they must stop at the remaining part

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minimumLength(s: String): Int {
    var i = 0; var j = s.lastIndex
    while (i < j && s[i] == s[j]) {
      while (i + 1 < j && s[i + 1] == s[j]) i++
      while (i < j - 1 && s[i] == s[j - 1]) j--
      i++; j--
    }
    return j - i + 1
  }

```
```rust

  pub fn minimum_length(s: String) -> i32 {
    let (mut i, mut j, s) = (0, s.len() - 1, s.as_bytes());
    while i < j && s[i] == s[j] {
      while i + 1 < j && s[i + 1] == s[j] { i += 1 }
      while i < j - 1 && s[i] == s[j - 1] { j -= 1 }
      i += 1; j -= 1
    }
    1 + (j - i) as i32
  }

```

