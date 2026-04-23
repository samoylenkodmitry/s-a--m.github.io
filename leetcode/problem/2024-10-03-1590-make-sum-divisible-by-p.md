---
layout: leetcode-entry
title: "1590. Make Sum Divisible by P"
permalink: "/leetcode/problem/2024-10-03-1590-make-sum-divisible-by-p/"
leetcode_ui: true
entry_slug: "2024-10-03-1590-make-sum-divisible-by-p"
---

[1590. Make Sum Divisible by P](https://leetcode.com/problems/make-sum-divisible-by-p/description/) medium
[blog post](https://leetcode.com/problems/make-sum-divisible-by-p/solutions/5863816/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03102024-1590-make-sum-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6MylEibVd_w)
![1.webp](/assets/leetcode_daily_images/e83a303e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/755

#### Problem TLDR

Min removed subarray length to make `remainder % p = 0` #medium #modulo

#### Intuition

Failed to solve this one.

The idea is: if we have a total `sum` and subarray sum `sub`, then `(sum - sub) % p == 0`:

```j

    // (sum-sub)%p==0
    // sum % p = sub % p

```
At this point I know, we should inspect the visited and awaited remainders, but exact solution still didn't clear to me.

Now, the part I didn't get myself:

```j

target = sum % p

target - sub % p == 0  <-- our condition

```

We visiting the prefix `sum` and inspecting the remainder `sum - target % p`.

#### Approach

* more time and more examples would help, you either see the math or don't
* steal someone else's solution

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minSubarray(nums: IntArray, p: Int): Int {
        val remToInd = HashMap<Long, Int>(); remToInd[0] = -1
        var ans = nums.size; var sum = 0L
        val target = nums.sumOf { it.toLong() % p } % p
        return nums.withIndex().minOf { (i, n) ->
            sum = (sum + n % p) % p
            remToInd[sum] = i
            i - (remToInd[(p + sum - target) % p] ?: -nums.size)
        }.takeIf { it < nums.size } ?: -1
    }

```
```rust

    pub fn min_subarray(nums: Vec<i32>, p: i32) -> i32 {
        let (mut ans, mut sum, mut wait) = (nums.len() as i32, 0, HashMap::new());
        wait.insert(0, -1);
        let target = (nums.iter().map(|&x| (x % p) as i64).sum::<i64>() % (p as i64)) as i32;
        let ans = nums.iter().enumerate().map(|(i, &n)| {
            sum = (sum + n % p) % p;
            wait.insert(sum, i as i32);
            let key = (p + sum - target) % p;
            if let Some(j) = wait.get(&key) { i as i32 - j } else { nums.len() as i32 }
        }).min().unwrap();
        if ans < nums.len() as i32 { ans } else { -1 }
    }

```
```c++

    int minSubarray(vector<int>& nums, int p) {
        std::unordered_map<long long, int> remToInd;
        remToInd[0] = -1; int ans = nums.size();
        long long sum = 0, target = 0;
        for (int num : nums) target = (target + num % p) % p;
        for (int i = 0; i < nums.size(); ++i) {
            sum = (sum + nums[i] % p) % p;
            remToInd[sum] = i; int key = (p + sum - target) % p;
            int diff = remToInd.count(key) ? i - remToInd[key] : nums.size();
            ans = std::min(ans, diff);
        }
        return ans < nums.size() ? ans : -1;
    }

```

