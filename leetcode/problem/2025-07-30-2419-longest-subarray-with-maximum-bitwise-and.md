---
layout: leetcode-entry
title: "2419. Longest Subarray With Maximum Bitwise AND"
permalink: "/leetcode/problem/2025-07-30-2419-longest-subarray-with-maximum-bitwise-and/"
leetcode_ui: true
entry_slug: "2025-07-30-2419-longest-subarray-with-maximum-bitwise-and"
---

[2419. Longest Subarray With Maximum Bitwise AND](https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/description/) medium
[blog post](https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/solutions/7023340/kotlin-rust-by-samoylenkodmitry-s9za/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30072025-2419-longest-subarray-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RwyDoaDPDNQ)
![1.webp](/assets/leetcode_daily_images/55be5618.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1065

#### Problem TLDR

Longest max OR subarray #medium #counting

#### Intuition

```j
    // 011
    // 010
    // 111
    // 100
```
Each new element decreases OR, consider only equal values.

#### Approach

* longest subarray of `max`es
* many one-liners possible
* 09/2024 - 13 minutes, 07/2025 - 10 minutes

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 43ms
    fun longestSubarray(n: IntArray, m: Int = n.max()) =
        n.runningFold(0) { l, x -> if (x == m) l + 1 else 0 }.max()

```
```kotlin

// 4ms
    fun longestSubarray(n: IntArray): Int {
        var m = 0; var r = 0; var l = 0
        for (x in n) if (x > m) { m = x; r = 1; l = 1 }
            else if (x < m) l = 0 else r = max(r, ++l)
        return r
    }

```
```rust

// 0ms
    pub fn longest_subarray(n: Vec<i32>) -> i32 {
        n.into_iter().dedup_with_count()
        .max_by_key(|&d| (d.1, d.0)).unwrap().0 as _
    }

```
```c++

// 1ms
    int longestSubarray(vector<int>& n) {
        int r = 0;
        for (int l = 0, m = 0; int x: n)
            x > m ? m = x, l = 1, r = 1 :
            x < m ? l = 0 : r = max(r, ++l);
        return r;
    }

```
```python3

// 36ms
    def longestSubarray(self, n: List[int]) -> int:
        m=max(n);return max(sum(1for _ in g) for x, g in groupby(n) if x==m)

```

