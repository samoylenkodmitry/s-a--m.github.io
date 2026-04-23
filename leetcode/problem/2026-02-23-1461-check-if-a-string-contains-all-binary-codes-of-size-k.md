---
layout: leetcode-entry
title: "1461. Check If a String Contains All Binary Codes of Size K"
permalink: "/leetcode/problem/2026-02-23-1461-check-if-a-string-contains-all-binary-codes-of-size-k/"
leetcode_ui: true
entry_slug: "2026-02-23-1461-check-if-a-string-contains-all-binary-codes-of-size-k"
---

[1461. Check If a String Contains All Binary Codes of Size K](https://open.substack.com/pub/dmitriisamoilenko/p/23022026-1461-check-if-a-string-contains?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/23022026-1461-check-if-a-string-contains?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23022026-1461-check-if-a-string-contains?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-JMK3yYBk2A)

![79e0f1f4-62cc-4ab3-b79c-6b7183b02def (1).webp](/assets/leetcode_daily_images/3347f649.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1278

#### Problem TLDR

All numbers as substring length of k #medium #sliding_window

#### Intuition

Use sliding window of k. Calculate the current number, add to a set. Uniqs count should be 2^k.

#### Approach

* use built-in windows, the k is small, can take substrings on the fly
* inverted solution: number of allowed duplicates is len-k+1 - 2^k
* return early

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 130ms
    fun hasAllCodes(s: String, k: Int) =
    s.windowed(k).toSet().size == 1 shl k
```
```rust
// 87ms
    pub fn has_all_codes(s: String, k: i32) -> bool {
        s.as_bytes().windows(k as usize).unique().count() as i32 == 1 << k
    }
```

