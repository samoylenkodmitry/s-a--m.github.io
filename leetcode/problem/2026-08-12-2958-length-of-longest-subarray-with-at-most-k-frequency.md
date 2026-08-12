---
layout: leetcode-entry
title: "2958. Length of Longest Subarray With at Most K Frequency"
permalink: "/leetcode/problem/2026-08-12-2958-length-of-longest-subarray-with-at-most-k-frequency/"
leetcode_ui: true
entry_slug: "2026-08-12-2958-length-of-longest-subarray-with-at-most-k-frequency"
---

[2958. Length of Longest Subarray With at Most K Frequency](https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/solutions/8456370/kotlin-rust-by-samoylenkodmitry-d6dj/) medium
[substack](https://dmitriisamoilenko.substack.com/p/12082026-2958-length-of-longest-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wc_x515LFFk)

https://dmitrysamoylenko.com/leetcode/

![12.08.2026.webp](/assets/leetcode_daily_images/12.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1449

#### Problem TLDR

Max subarray freq less than K

#### Intuition

Sliding window: always slide forward, shring while its bad freq.
Max window: increase if good, move right if bad

#### Approach

Kotlin: HashMap merge(x, 1, Int::plus)
Rust: HashMap entry and_modify

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxSubarrayLength(n: IntArray, k: Int)=run {
        val f = HashMap<Int, Int>(); var j = 0; var bad = 0
        for (i in n.indices) {
            if (f.merge(n[i], 1, Int::plus)!! == k + 1) bad++
            if (bad > 0 && f.merge(n[j++], -1, Int::plus) == k) bad--
        }; n.size - j
    }
```
```rust
    pub fn max_subarray_length(n: Vec<i32>, k: i32) -> i32 {
        let (mut f, mut j, mut b) = (HashMap::new(), 0, 0);
        for &x in &n {
            f.entry(x).and_modify(|c| { b += (*c == k) as i32; *c += 1 }).or_insert(1);
            if b > 0 {
                f.entry(n[j]).and_modify(|c| { b -= (*c == k + 1) as i32; *c -= 1 });
                j += 1;
            }
        } (n.len() - j) as i32
    }
```

