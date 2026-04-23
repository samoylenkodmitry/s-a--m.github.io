---
layout: leetcode-entry
title: "409. Longest Palindrome"
permalink: "/leetcode/problem/2024-06-04-409-longest-palindrome/"
leetcode_ui: true
entry_slug: "2024-06-04-409-longest-palindrome"
---

[409. Longest Palindrome](https://leetcode.com/problems/longest-palindrome/description/) easy
[blog post](https://leetcode.com/problems/longest-palindrome/solutions/5255875/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04062024-409-longest-palindrome?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rFm-0gOSYXc)
![2024-06-04_07-01_1.webp](/assets/leetcode_daily_images/231a47f8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/628

#### Problem TLDR

Max palindrome length from chars #easy

#### Intuition

Don't mistaken this problem with `find the longest palindrome`, because this time we need to `build` one. (I have spent 5 minutes solving the wrong problem)

To build a palindrome, we need `even` counts of chars and `at most` one `odd`.

#### Approach

* we can use `groupBy`, `sumBy` and `any`
* `f & 1` operation will convert any `odd` number into `1`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, but O(n) for the `groupBy` solution, which can be optimized

#### Code

```kotlin

    fun longestPalindrome(s: String): Int =
        s.groupBy { it }.values.run {
            2 * sumBy { it.size / 2 } +
            if (any { it.size % 2 > 0 }) 1 else 0
        }

```
```rust

    pub fn longest_palindrome(s: String) -> i32 {
        let (mut freq, mut res, mut o) = (vec![0;128], 0, 0);
        for b in s.bytes() { freq[b as usize] += 1 }
        for f in freq { o |= f & 1; res += f / 2 }
        2 * res + o
    }

```

