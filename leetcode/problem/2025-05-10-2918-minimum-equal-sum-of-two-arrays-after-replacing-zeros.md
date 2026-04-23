---
layout: leetcode-entry
title: "2918. Minimum Equal Sum of Two Arrays After Replacing Zeros"
permalink: "/leetcode/problem/2025-05-10-2918-minimum-equal-sum-of-two-arrays-after-replacing-zeros/"
leetcode_ui: true
entry_slug: "2025-05-10-2918-minimum-equal-sum-of-two-arrays-after-replacing-zeros"
---

[2918. Minimum Equal Sum of Two Arrays After Replacing Zeros](https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/description/) medium
[blog post](https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/solutions/6730634/kotlin-rust-by-samoylenkodmitry-udm7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10052025-2918-minimum-equal-sum-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tTCD1ZB3FLU)
![1.webp](/assets/leetcode_daily_images/9352c668.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/984

#### Problem TLDR

Min equal array sum, fill zeros #medium

#### Intuition

Any zero place can act as any number. Compare minimum sums by filling zeros with ones, then equalize.

#### Approach

* the interesting golf is how to make it CPU-branchless: `if (x == 0) 1 else 0` can be written as `((x | -x) >> 31) + 1` where `x | -x` would will everything but a sign bit, transforming into -1 for any non-zero value

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

|Kotlin|Rust|C++|
|------|----|---|
|![image.png](/assets/leetcode_daily_images/21c3e7ea.webp){:style="width:100px"}|![image.png](/assets/leetcode_daily_images/07a3c43d.webp){:style="width:100px"}|![image.png](/assets/leetcode_daily_images/88e8a85b.webp){:style="width:100px"}|

```kotlin

// 523ms
    fun minSum(n1: IntArray, n2: IntArray): Long {
        val s1 = n1.sumOf { 1L * max(1, it) }; val s2 = n2.sumOf { 1L * max(1, it) }
        return if (s1 < s2 && 0 !in n1 || s1 > s2 && 0 !in n2) -1 else max(s1, s2)
    }

```
```kotlin

// 411ms https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/submissions/1630261946
    fun minSum(nums1: IntArray, nums2: IntArray): Long {
        var s1 = 0L; var s2 = 0L; var z1 = 0; var z2 = 0
        for (x in nums1) { s1 += x + ((x or -x).ushr(31) xor 1); z1 = z1 or ((x or -x).ushr(31) xor 1) }
        for (x in nums2) { s2 += x + ((x or -x).ushr(31) xor 1); z2 = z2 or ((x or -x).ushr(31) xor 1) }
        return if (s1 < s2 && z1 < 1 || s1 > s2 && z2 < 1) -1 else max(s1, s2)
    }

```
```rust

// 8ms https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/submissions/1630002606
    pub fn min_sum(n1: Vec<i32>, n2: Vec<i32>) -> i64 {
        let s1: i64 = n1.iter().map(|&n| n.max(1) as i64).sum();
        let s2: i64 = n2.iter().map(|&n| n.max(1) as i64).sum();
        let z1 = n1.contains(&0); let z2 = n2.contains(&0);
        if (s1 < s2 && !z1) || (s1 > s2 && !z2) { -1 } else { s1.max(s2) }
    }

```
```c++

// 43ms https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/submissions/1630272912
    long long minSum(vector<int>& n1, vector<int>& n2) {
        long long s1 = size(n1), s2 = size(n2); int z1 = s1, z2 = s2;
        for (int& n: n1) s1 += n + ((n|-n)>>31), z1 += (n|-n)>>31;
        for (int& n: n2) s2 += n + ((n|-n)>>31), z2 += (n|-n)>>31;
        return s1 < s2 && !z1 || s1 > s2 && !z2 ? -1 : max(s1, s2);
    }

```

