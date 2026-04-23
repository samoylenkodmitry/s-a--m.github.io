---
layout: leetcode-entry
title: "2799. Count Complete Subarrays in an Array"
permalink: "/leetcode/problem/2025-04-24-2799-count-complete-subarrays-in-an-array/"
leetcode_ui: true
entry_slug: "2025-04-24-2799-count-complete-subarrays-in-an-array"
---

[2799. Count Complete Subarrays in an Array](https://leetcode.com/problems/count-complete-subarrays-in-an-array/description/) medium
[blog post](https://leetcode.com/problems/count-complete-subarrays-in-an-array/solutions/6682469/kotlin-rust-by-samoylenkodmitry-mnqn/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24042025-2799-count-complete-subarrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_gn_JSLwAxc)
![1.webp](/assets/leetcode_daily_images/e28cb242.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/968

#### Problem TLDR

Subarrays having all uniqs #medium #two_pointers

#### Intuition

This is a standard two-pointers problem: count frequencies, always move right, move left until condition. All prefixes are valid starts of the subarrays.

```j

    // 0 1 2 3 4
    // 1,3,1,2,2
    //     j   i

```

* subarrays are valid for all indexes `0..j`

#### Approach

* try to dry-run solution before pressing "submit"

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countCompleteSubarrays(nums: IntArray): Int {
        val f = IntArray(2001); var u = nums.toSet().size
        return nums.sumOf { x ->
            if (f[x]++ < 1) u--
            while (u < 1) if (--f[nums[f[0]++]] < 1) u++
            f[0]
        }
    }

```
```kotlin

    fun countCompleteSubarrays(nums: IntArray): Int {
        val f = IntArray(2001); var j = 0; var u = 0
        for (x in nums) if (f[x] < 1) { u++; ++f[x] }
        return nums.sumOf { x ->
            if (f[x]++ < 2) u--
            while (u < 1) if (--f[nums[j++]] < 2) u++
            j
        }
    }

```
```kotlin

    fun countCompleteSubarrays(nums: IntArray): Int {
        val f = IntArray(2001); var j = 0; var r = 0
        for (x in nums) {
            if (f[x]++ < 1) r = 0
            while (f[nums[j]] > 1) --f[nums[j++]]
            r += j + 1
        }
        return r
    }

```
```rust

    pub fn count_complete_subarrays(nums: Vec<i32>) -> i32 {
        let (mut f, mut j, mut u) = ([0; 2001], 0, 0);
        for &x in &nums { if f[x as usize] < 1 { u += 1; f[x as usize] = 1}}
        (0..nums.len()).map(|i| { let x = nums[i] as usize;
            if f[x] < 2 { u -= 1 }; f[x] += 1;
            while u < 1 { let x = nums[j] as usize; j += 1;
                f[x] -= 1; if f[x] < 2 { u += 1 }}
            j
        }).sum::<usize>() as _
    }

```
```c++

    int countCompleteSubarrays(vector<int>& nums) {
        int f[2001] = {}, u = 0, r = 0, j = 0;
        for (int x: nums) if (!f[x]) ++u, ++f[x];
        for (int x: nums) {
            if (f[x]++ < 2) --u;
            while (u < 1) if (--f[nums[j++]] < 2) ++u;
            r += j;
        } return r;
    }

```

