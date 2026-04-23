---
layout: leetcode-entry
title: "3392. Count Subarrays of Length Three With a Condition"
permalink: "/leetcode/problem/2025-04-27-3392-count-subarrays-of-length-three-with-a-condition/"
leetcode_ui: true
entry_slug: "2025-04-27-3392-count-subarrays-of-length-three-with-a-condition"
---

[3392. Count Subarrays of Length Three With a Condition](https://leetcode.com/problems/count-subarrays-of-length-three-with-a-condition/description/) easy
[blog post](https://leetcode.com/problems/count-subarrays-of-length-three-with-a-condition/solutions/6691621/kotlin-rust-by-samoylenkodmitry-ft2i/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27042025-3392-count-subarrays-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yxyvTHb9mRA)
![1.webp](/assets/leetcode_daily_images/88a2f5ca.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/971

#### Problem TLDR

3-subarrays 2a + 2c == b #easy

#### Intuition

Constrains are small, even the brute-force solution is O(n)

#### Approach

* let's golf it
* some CPU-cache friendliness possible

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 31ms
    fun countSubarrays(n: IntArray) = n.asList()
    .windowed(3).count { 2 * it.sum() == 3 * it[1] }

```
```kotlin

// 2ms
    fun countSubarrays(n: IntArray): Int {
        var c = 0; var l = 0; var m = -300
        for (r in n) {
            if (l + l + r + r == m) c++
            l = m; m = r
        }
        return c
    }

```
```rust

// 0ms
    pub fn count_subarrays(n: Vec<i32>) -> i32 {
        n[..].windows(3).filter(|w| 2 * w[0] + 2 * w[2] == w[1]).count() as _
    }

```
```c++

// 0ms
    int countSubarrays(vector<int>& n) {
        int c = 0, l = 0, m = 300;
        for (int r: n) c += l + l + r + r == m, l = m, m = r;
        return c;
    }

```

