def removeDuplicates(nums) -> int:
    nums_list = list()

    for num in nums:
        if num in nums_list:
            continue
        else:
            nums_list.append(num)

    nums_list.extend(['_' for _ in range(len(nums) - len(nums_list))])
    print(nums_list)


nums = [1, 1, 2, 2, 3, 4, 4, 5]
removeDuplicates(nums)
