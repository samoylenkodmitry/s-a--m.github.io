---
layout: leetcode-entry
title: "3151. Special Array I"
permalink: "/leetcode/problem/2025-02-01-3151-special-array-i/"
leetcode_ui: true
entry_slug: "2025-02-01-3151-special-array-i"
---

[3151. Special Array I](https://leetcode.com/problems/special-array-i/description/) easy
[blog post](https://leetcode.com/problems/special-array-i/solutions/6356313/kotlin-rust-by-samoylenkodmitry-ognc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01022025-3151-special-array-i?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/F4Q1sUavzzg)
![1.webp](/assets/leetcode_daily_images/fff2b551.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/882

#### Problem TLDR

All siblings even-odd #easy

#### Intuition

Let's golf

#### Approach

* there is also a bitmask solution for i128 ints: only two masks possible `010101...` and `101010...`
* can you make it shorter?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$ O(n) for Kotlin golf

#### Code

```kotlin

    fun isArraySpecial(nums: IntArray) =
        Regex("0, 0|1, 1") !in "${nums.map { it % 2 }}"

```
```rust

    pub fn is_array_special(nums: Vec<i32>) -> bool {
        (1..nums.len()).all(|i| nums[i] % 2 != nums[i - 1] % 2)
    }

```
```c++

    bool isArraySpecial(vector<int>& n) {
        int r = 1; for(int i = size(n); --i; r &= n[i] % 2 ^ n[i - 1] % 2); return r;
    }

```

