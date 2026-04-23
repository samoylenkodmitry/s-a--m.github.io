---
layout: leetcode-entry
title: "2348. Number of Zero-Filled Subarrays"
permalink: "/leetcode/problem/2025-08-19-2348-number-of-zero-filled-subarrays/"
leetcode_ui: true
entry_slug: "2025-08-19-2348-number-of-zero-filled-subarrays"
---

[2348. Number of Zero-Filled Subarrays](https://leetcode.com/problems/number-of-zero-filled-subarrays/description) medium
[blog post](https://leetcode.com/problems/number-of-zero-filled-subarrays/solutions/7098501/kotlin-rust-by-samoylenkodmitry-oucq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19082025-2348-number-of-zero-filled?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cm29MlKQnQs)

![1.webp](/assets/leetcode_daily_images/7c391bbf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1086

#### Problem TLDR

Count 0-subarrays #medium #counting

#### Intuition

Count zero islands, use arithmetic sum: `n(n+1)/2`

#### Approach

* instead of arithmetics, just add current count to the result

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 5ms
    fun zeroFilledSubarray(n: IntArray) =
        n.fold(0L) { r, t ->
            if (t == 0) ++n[0] else n[0] = 0; r + n[0]
        }

```
```kotlin

// 3ms
    fun zeroFilledSubarray(n: IntArray): Long {
        var res = 0L; var curr = 0
        for (x in n) {
            if (x == 0) ++curr else curr = 0
            res += curr
        }
        return res
    }

```
```rust

// 0ms
    pub fn zero_filled_subarray(n: Vec<i32>) -> i64 {
        let (mut r, mut c) = (0, 0);
        for x in n { if (x == 0) { c += 1 } else { c = 0 }; r += c } r
    }

```
```c++

// 0ms
    long long zeroFilledSubarray(vector<int>& n) {
        long long r = 0;
        for (int c = 0; int x: n) r += c = x ? 0: ++c;
        return r;
    }

```
```python

// 49ms
    zeroFilledSubarray = lambda _,n: sum(accumulate(n, lambda c,x: (c+1)*(x==0), initial=0))

```

