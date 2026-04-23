---
layout: leetcode-entry
title: "241. Different Ways to Add Parentheses"
permalink: "/leetcode/problem/2024-09-19-241-different-ways-to-add-parentheses/"
leetcode_ui: true
entry_slug: "2024-09-19-241-different-ways-to-add-parentheses"
---

[241. Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/description/) medium
[blog post](https://leetcode.com/problems/different-ways-to-add-parentheses/solutions/5807534/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19092024-241-different-ways-to-add?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bsa7yz36XTI)
![1.webp](/assets/leetcode_daily_images/f65bb9a4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/740

#### Problem TLDR

Eval all possible parenthesis placements #medium #dynamic_programming

#### Intuition

This problem is all about splitting the task into a subproblem. Let's make a `tree` where each node is the `operation` on it's `left` and `right` subtree.

Now, first compute left and right result, then invoke an operation for each operation in the current expression.

#### Approach

* memoization is not necessary
* if there is no operations, then expression is a single number

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin

    fun diffWaysToCompute(expression: String): List<Int> = buildList {
        for ((i, c) in expression.withIndex()) if (!c.isDigit()) {
            val leftList = diffWaysToCompute(expression.take(i))
            val rightList = diffWaysToCompute(expression.drop(i + 1))
            for (left in leftList) for (right in rightList) add(when (c) {
                '+' -> left + right
                '-' -> left - right
                else -> left * right
            })
        }
        if (isEmpty()) add(expression.toInt())
    }

```
```rust

    pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
        let (mut i, mut res) = (0, vec![]);
        for i in 0..expression.len() {
            let b = expression.as_bytes()[i];
            if let b'+' | b'-' | b'*' = b {
                let left_res = Self::diff_ways_to_compute(expression[..i].to_string());
                let right_res = Self::diff_ways_to_compute(expression[i + 1..].to_string());
                for left in &left_res { for right in &right_res { res.push(match b {
                    b'+' => left + right, b'-' => left - right, _ => left * right
                })}}
            }}
        if res.is_empty() { vec![expression.parse::<i32>().unwrap()] } else { res }
    }

```
```c++

    vector<int> diffWaysToCompute(string expression) {
        vector<int> res;
        for (int i = 0; i < expression.size(); i++) {
            auto c = expression[i];
            if (c == '+' || c == '-' || c == '*') {
                vector<int> left_res = diffWaysToCompute(expression.substr(0, i));
                vector<int> right_res = diffWaysToCompute(expression.substr(i + 1, expression.size() - i - 1));
                for (auto left: left_res) for (auto right: right_res)
                    res.push_back(c == '+' ? left + right : c == '-' ? left - right : left * right);
            }
        }
        if (res.size() == 0) res.push_back(std::stoi(expression));
        return res;
    }

```

