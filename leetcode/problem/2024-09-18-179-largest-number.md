---
layout: leetcode-entry
title: "179. Largest Number"
permalink: "/leetcode/problem/2024-09-18-179-largest-number/"
leetcode_ui: true
entry_slug: "2024-09-18-179-largest-number"
---

[179. Largest Number](https://leetcode.com/problems/largest-number/description/) medium
[blog post](https://leetcode.com/problems/largest-number/solutions/5803278/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18092024-179-largest-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aes3e4Az3To)
![1.webp](/assets/leetcode_daily_images/b8580999.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/739

#### Problem TLDR

Concatenate nums to largest number #medium #math

#### Intuition

The intuition was and still is hard for me.
(My own *wrong* intuition is that we can only do the backtracking and a full search)

Assuming we have to choose between `3 30 32`, we should compare `3_30`, `3_32`, `30_3`, `30_32`, `32_3`, `32_30`, and choose - `3_32`.

For proving the correctness of applying the `sorting` I would pass to @DBabichev https://leetcode.com/problems/largest-number/solutions/863489/python-2-lines-solution-using-sort-explained/

(You have to prove the transtivity if (a > b), and (b > c) then (a > c) https://en.wikipedia.org/wiki/Comparison_sort)

```j

    // 3 9 90
    // 9093 9903 9390
    //
    // 3 30 34
    //      *
    // *
    //   *

    // 3 30 32
    // *
    //      *
    //   *

    // 31 310 312
    // *
    //        *
    //    *

```

#### Approach

* we can convert to string before or after the sorting
* the "0" corner case can be fixed by checking the first number

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun largestNumber(nums: IntArray) = nums
        .sortedWith { a, b -> "$b$a".compareTo("$a$b") }
        .joinToString("").takeIf { it[0] != '0' } ?: "0"

```
```rust

    pub fn largest_number(mut nums: Vec<i32>) -> String {
        nums.sort_by(|a, b| {
            let (a, b) = (format!("{b}{a}"), format!("{a}{b}")); a.cmp(&b)});
        if nums[0] == 0 { return "0".into() }
        nums.into_iter().map(|n| n.to_string()).collect()
    }

```
```c++

    string largestNumber(vector<int>& nums) {
        std::sort(nums.begin(), nums.end(), [](int a, int b){
            return "" + to_string(b) + to_string(a) < "" + to_string(a) + to_string(b);
        });
        string res; for (auto n: nums) res += to_string(n);
        return res[0] == '0' ? "0": res;
    }

```

