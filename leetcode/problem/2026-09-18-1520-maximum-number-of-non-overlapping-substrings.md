---
layout: leetcode-entry
title: "1520. Maximum Number of Non-Overlapping Substrings"
permalink: "/leetcode/problem/2026-09-18-1520-maximum-number-of-non-overlapping-substrings/"
leetcode_ui: true
entry_slug: "2026-09-18-1520-maximum-number-of-non-overlapping-substrings"
---

[1520. Maximum Number of Non-Overlapping Substrings](https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings/solutions/8527783/kotlin-rust-by-samoylenkodmitry-m70d/) hard
[substack](https://dmitriisamoilenko.substack.com/p/18092026-1520-maximum-number-of-non?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/k_7uq89gTTk)

https://dmitrysamoylenko.com/leetcode/

![18.09.2026.webp](/assets/leetcode_daily_images/18.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1486

#### Problem TLDR

Max substrings containing all it's chars

#### Intuition

For each letter expand substring to the left and to the right.
Scan the string, take each end of the interval and make a choice dp[start-1]+current vs dp[i-1].
Another way: take each start and grow to the right as you scan the string. Take substring if it improves the prevous choice.
Another way: sort the intervals. Take the leftmost shortest first.

#### Approach

* Rust: !0 is usize::MAX

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxNumOfSubstrings(s: String): List<String> {
        val L = IntArray(26) { s.indexOf('a' + it) }
        val R = IntArray(26) { s.lastIndexOf('a' + it) }
        var end = -1
        return (0..25).filter { L[it] >= 0 }.mapNotNull {
            var r = R[it]; var j = L[it]
            while (j <= r && L[s[j] - 'a'] >= L[it]) r = maxOf(r, R[s[j++] - 'a'])
            if (j > r) L[it]..r else null
        }.sortedWith(compareBy({ it.last }, { -it.first })).mapNotNull {
            if (it.first > end) s.substring(it).also { _ -> end = it.last } else null
        }
    }
```
```rust
    pub fn max_num_of_substrings(s: String) -> Vec<String> {
        let (b, mut L, mut R) = (s.as_bytes(), [!0; 26], [0; 26]);
        let c = |i| (b[i] - b'a') as usize; let (mut res, mut last) = (vec![], !0);
        b.iter().enumerate().for_each(|(i, _)| { L[c(i)] = L[c(i)].min(i); R[c(i)] = i });
        for i in 0..b.len() {
            if i != L[c(i)] { continue }; let (mut r, mut j) = (R[c(i)], i);
            while j <= r && L[c(j)] >= i { r = r.max(R[c(j)]); j += 1 }
            if j > r && (last == !0 || i > last || r < last) {
                if last != !0 && r < last { res.pop(); }
                res.push(s[i..=r].into()); last = r
            }
        } res
    }
```

